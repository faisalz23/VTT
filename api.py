# api.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from summarizer import summarize_text

# Load env
load_dotenv()

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

        summary = summarize_text(text)
        return jsonify({"summary": summary})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
