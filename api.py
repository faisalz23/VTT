import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "connected", "message": "Backend is running"})

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Teks kosong"}), 400

        prompt = f"""
Anda adalah seorang dokter patologi berpengalaman. Buat ringkasan/impresi patologi klinis yang rapi, terstruktur, dan ringkas sesuai standar laporan.

Kebijakan istilah:
- Gunakan dan/atau standarkan terminologi medis internasional (bahasa Inggris) untuk nama penyakit, prosedur, marker/imunohistokimia, gen, stadium, dan obat.
- Jangan terjemahkan singkatan medis (mis. ER, PR, HER2, EGFR, Ki-67). Koreksi ejaan istilah medis bila perlu.
- Bila sumber berbahasa Indonesia, tetap tampilkan padanan Inggris-nya di dalam tanda kurung jika relevan. Contoh: "karsinoma duktal (ductal carcinoma)".
- Pertahankan satuan/angka persentase/derajat sesuai kaidah.

Format keluaran yang disarankan (jika cocok):
- Temuan Utama:
- Diagnosis/Impresi:
- Rekomendasi/Tindak Lanjut:

Teks sumber:
{text}

Ringkasan:
"""

        # Panggil Groq API
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.3
        )

        summary = response.choices[0].message.content.strip()

        if "</think>" in summary:
            summary = summary.split("</think>", 1)[1].strip()

        return jsonify({"summary": summary})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
