import os
import time
import re
from threading import Event

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from groq import Groq
from dotenv import load_dotenv
from werkzeug.exceptions import HTTPException

# =========================
# Config & Init
# =========================
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

stop_flags = {}  # {sid: Event}

# =========================
# Helpers
# =========================
def strip_think(text: str) -> str:
    """Hapus blok <think>...</think> bila ada."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def build_prompt(text: str) -> str:
    return f"""
Anda adalah seorang dokter patologi berpengalaman.
Langsung berikan ringkasan final saja, tanpa proses berpikir.
Ikuti format:

**Ringkasan Patologi Klinis**

**Deskripsi Makroskopis:**
- ...

**Diagnosis / Impresi:**
- ...

**Rekomendasi / Tindak Lanjut:**
- ...

Aturan ketat:
- Hanya ekstrak fakta yang ada pada teks sumber.
- Pertahankan angka/satuan persis seperti tertulis.
- Jangan menambah atau mengubah fakta yang tidak ada di teks.

Teks sumber:
{text}

Ringkasan:
"""

def _parse_retry_after_seconds(message: str):
    try:
        m = re.search(r"in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", message)
        if not m:
            return None
        minutes = float(m.group(1)) if m.group(1) else 0.0
        seconds = float(m.group(2))
        return minutes * 60.0 + seconds
    except Exception:
        return None

# =========================
# Error handler global
# =========================
@app.errorhandler(Exception)
def handle_exception(e):
    code = 500
    msg = str(e)
    if isinstance(e, HTTPException):
        code = e.code or 500
        msg = e.description
    return jsonify({"error": msg}), code

# =========================
# Routes (Pages)
# =========================
@app.route("/")
def dashboard():
    """Halaman dashboard (dari snippet kedua)."""
    return render_template("dashboard.html")

@app.route("/voice")
def voice_page():
    """Halaman voice-to-text (dari snippet pertama)."""
    return render_template("index.html")

@app.route("/history")
def history_page():
    """Halaman riwayat (history)."""
    return render_template("history.html")

# NOTE:
# Jika Anda ingin index.html di root "/", tukar saja:
#   - Ganti fungsi dashboard() untuk merender index.html
#   - Buat route lain untuk dashboard, misal "/dashboard"

# =========================
# Routes (APIs)
# =========================
@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "connected", "message": "Backend is running"})

# ---------- HTTP summarize ----------
@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Teks kosong"}), 400

        prompt = build_prompt(text)
        print("[/summarize] text_len=", len(text))

        max_retries = 3
        base_sleep = 3.0
        attempt = 0
        while True:
            try:
                resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=MODEL,
                    temperature=0.3,
                )
                summary_raw = (resp.choices[0].message.content or "").strip()
                summary = strip_think(summary_raw)
                return jsonify({"summary": summary})
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print("[/summarize] ERROR:", msg)
                low = str(e).lower()
                is_rate = "rate limit" in low or "rate_limit" in low
                is_conn = any(k in low for k in ["connection", "timeout", "timed out", "temporarily"])
                retry_after = _parse_retry_after_seconds(str(e)) or base_sleep
                attempt += 1

                if (is_rate or is_conn) and attempt <= max_retries:
                    sleep_for = retry_after * (2 ** (attempt - 1))
                    print(f"[/summarize] retry in {sleep_for:.1f}s (attempt {attempt}/{max_retries})")
                    time.sleep(sleep_for)
                    continue

                if is_rate:
                    return jsonify({"error": "rate_limit", "message": str(e), "retry_after": max(5, int(retry_after))}), 429
                if is_conn:
                    return jsonify({"error": "upstream_connection", "message": str(e)}), 502
                return jsonify({"error": str(e)}), 500

    except Exception as e:
        print("ERROR /summarize (outer):", f"{type(e).__name__}: {e}")
        return jsonify({"error": str(e)}), 500

# ---------- STREAM summarize ----------
@socketio.on("summarize_stream")
def handle_summarize_stream(data):
    sid = request.sid
    text = (data.get("text") or "").strip()
    if not text:
        emit("summary_stream", {"error": "Teks kosong"})
        return

    prompt = build_prompt(text)
    print(f"[stream] start SID={sid} text_len={len(text)}")

    stop_evt = Event()
    stop_flags[sid] = stop_evt

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            temperature=0.3,
            stream=True
        )

        token_count = 0
        collected = []
        for chunk in response:
            if stop_evt.is_set():
                print(f"[stream] stopped by client SID={sid}")
                break

            try:
                choice = chunk.choices[0]
            except Exception:
                continue

            text_piece = None
            delta = getattr(choice, "delta", None)
            if delta and getattr(delta, "content", None):
                text_piece = delta.content
            if not text_piece:
                message_obj = getattr(choice, "message", None)
                if message_obj and getattr(message_obj, "content", None):
                    text_piece = message_obj.content

            if text_piece:
                token_count += len(text_piece)
                collected.append(text_piece)
                emit("summary_stream", {"token": text_piece})

        final_raw = "".join(collected).strip()
        final_fmt = strip_think(final_raw)
        emit("summary_stream", {"final": final_fmt, "end": True})
        print(f"[stream] end SID={sid} tokens={token_count}")

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"[stream] error SID={sid}: {msg}")
        emit("summary_stream", {"error": str(e)})
    finally:
        stop_flags.pop(sid, None)

@socketio.on("stop_stream")
def handle_stop_stream():
    sid = request.sid
    if sid in stop_flags:
        stop_flags[sid].set()
    emit("stop_stream")

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    if sid in stop_flags:
        stop_flags[sid].set()
    print(f"[socket] disconnect SID={sid}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Gunakan SocketIO untuk menjalankan server (bukan app.run)
    socketio.run(app, debug=True, use_reloader=False,
                 host="127.0.0.1", port=int(os.environ.get("PORT", 5001)))
