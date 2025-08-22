# summarizer.py
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def summarize_text(text: str) -> str:
    if not text.strip():
        return ""

    prompt = f"""
Anda adalah seorang dokter patologi berpengalaman. Buat ringkasan/impresi patologi klinis yang rapi, terstruktur, dan ringkas sesuai standar laporan.

Kebijakan istilah:
- Gunakan dan/atau standarkan terminologi medis internasional (bahasa Inggris) untuk nama penyakit, prosedur, marker/imunohistokimia, gen, stadium, dan obat.
- Jangan terjemahkan singkatan medis (mis. ER, PR, HER2, EGFR, Ki-67). Koreksi ejaan istilah medis bila perlu.
- Bila sumber berbahasa Indonesia, tetap tampilkan padanan Inggris-nya di dalam tanda kurung jika relevan.
- Pertahankan satuan/angka persentase/derajat sesuai kaidah.

Format keluaran yang disarankan (jika cocok):
- Temuan Utama:
- Diagnosis/Impresi:
- Rekomendasi/Tindak Lanjut:

Teks sumber:
{text}

Ringkasan:
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        summary = response.choices[0].message.content.strip()

        # Buang <think>...</think> kalau ada
        if "</think>" in summary:
            summary = summary.split("</think>", 1)[1].strip()

        return summary
    except Exception as e:
        return f"Error during summarization: {e}"
