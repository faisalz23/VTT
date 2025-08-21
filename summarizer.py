# # summarizer.py
# import os
# import re
# import json
# import time
# import logging
# from typing import Optional

# import requests
# import torch

# # Try to import transformers objects only when attempting local load
# try:
#     from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#     TRANSFORMERS_AVAILABLE = True
# except Exception:
#     TRANSFORMERS_AVAILABLE = False

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class TextSummarizer:
#     """
#     Summarizer that prefers local MedGemma-4b if available; otherwise uses Hugging Face Inference API.
#     For HF Inference API you must set environment variable HUGGINGFACE_API_KEY with a valid token.
#     """
#     def __init__(self, model_name: str = "google/medgemma-27b-it"):
#         self.model_name = model_name
#         self.device = 0 if torch.cuda.is_available() else -1
#         self.local_pipe = None
#         self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY", None)
#         self.hf_endpoint = f"https://api-inference.huggingface.co/models/{self.model_name}"
#         self._try_load_local()

#     def _try_load_local(self):
#         """Attempt to load local model with transformers.pipeline (may fail if resource/transformers not available)."""
#         if not TRANSFORMERS_AVAILABLE:
#             logger.info("transformers not available — will use Hugging Face Inference API if token present.")
#             return

#         try:
#             logger.info(f"Attempting to load local model {self.model_name} (device={'cuda' if self.device == 0 else 'cpu'}) ...")
#             tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
#             model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name,
#                 torch_dtype=torch.bfloat16 if (torch.cuda.is_available()) else torch.float32,
#                 device_map="auto" if torch.cuda.is_available() else None,
#             )
#             # use text-generation pipeline for decoder-only LLM
#             self.local_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=self.device)
#             logger.info("Local model loaded successfully.")
#         except Exception as e:
#             logger.exception(f"Local model load failed: {e}")
#             self.local_pipe = None

#     def _call_hf_api(self, prompt: str, max_new_tokens: int = 150, timeout: int = 120) -> str:
#         """Call Hugging Face Inference API (text generation). Requires HUGGINGFACE_API_KEY env var."""
#         if not self.hf_api_key:
#             raise RuntimeError("Hugging Face API key is not set (HUGGINGFACE_API_KEY).")

#         headers = {"Authorization": f"Bearer {self.hf_api_key}", "Content-Type": "application/json"}
#         payload = {
#             "inputs": prompt,
#             "options": {"use_cache": False, "wait_for_model": True},
#             "parameters": {
#                 "max_new_tokens": max_new_tokens,
#                 "temperature": 0.2,
#                 "top_p": 0.95,
#                 "repetition_penalty": 1.05,
#                 "do_sample": False
#             }
#         }

#         resp = requests.post(self.hf_endpoint, headers=headers, json=payload, timeout=timeout)
#         if resp.status_code != 200:
#             logger.error("HF API error: %s %s", resp.status_code, resp.text)
#             raise RuntimeError(f"Hugging Face API error {resp.status_code}: {resp.text}")

#         data = resp.json()
#         # HF generation returns list with 'generated_text' or sometimes a dict with detail
#         if isinstance(data, dict) and data.get("error"):
#             raise RuntimeError(f"Hugging Face API returned error: {data['error']}")
#         # expected: [{'generated_text': '...'}]
#         generated = ""
#         try:
#             if isinstance(data, list) and "generated_text" in data[0]:
#                 generated = data[0]["generated_text"]
#             elif isinstance(data, dict) and "generated_text" in data:
#                 generated = data["generated_text"]
#             else:
#                 # some models return different shape; try to str() fallback
#                 generated = str(data)
#         except Exception:
#             generated = str(data)
#         return generated

#     def summarize(self, text: str, max_length: int = 150) -> str:
#         """
#         Summarize text (prefer local pipe; fallback to Hugging Face Inference API; final fallback is simple extractive).
#         max_length = number of tokens/words to generate approximately (passed as max_new_tokens or similar)
#         """
#         if not text or not text.strip():
#             return ""

#         # normalize whitespace
#         text = re.sub(r"\s+", " ", text).strip()

#         prompt = (
#             "Anda adalah asisten yang bertugas merangkum teks transkrip percakapan atau dokumen dalam bahasa Indonesia. "
#     "Fokuslah pada inti informasi, hapus pengulangan, dan sampaikan dengan bahasa yang jelas, ringkas, "
#     "dan mudah dipahami. Gunakan paragraf singkat atau poin-poin jika diperlukan.\n\n"
#     f"Teks:\n{text}\n\nRingkasan:"
#         )

#         # 1) Try local pipeline if loaded
#         if self.local_pipe is not None:
#             try:
#                 out = self.local_pipe(prompt, max_new_tokens=max_length, do_sample=False, temperature=0.2)
#                 # out is typically list of dicts with 'generated_text'
#                 generated = out[0].get("generated_text") if isinstance(out, list) else str(out)
#                 # Extract portion after "Ringkasan:" if present
#                 if "Ringkasan:" in generated:
#                     return generated.split("Ringkasan:")[-1].strip()
#                 # else try best-effort
#                 return generated.strip()
#             except Exception as e:
#                 logger.exception("Local generation failed, will fallback to HF API: %s", e)
#                 self.local_pipe = None  # disable local for future tries

#         # 2) Try HF Inference API
#         try:
#             generated = self._call_hf_api(prompt, max_new_tokens=max_length)
#             if "Ringkasan:" in generated:
#                 return generated.split("Ringkasan:")[-1].strip()
#             return generated.strip()
#         except Exception as e:
#             logger.exception("Hugging Face API call failed: %s", e)

#         # 3) Last-resort simple extractive summarization
#         return self._basic_extractive(text, max_length)

#     def _basic_extractive(self, text: str, max_chars: int = 200) -> str:
#         # Very simple extractive: take top sentences by word frequency score
#         sentences = re.split(r'(?<=[\.\!\?])\s+', text)
#         if not sentences:
#             return text[:max_chars] + ("..." if len(text) > max_chars else "")
#         words = re.findall(r"\w+", text.lower())
#         freq = {}
#         for w in words:
#             if len(w) <= 2:
#                 continue
#             freq[w] = freq.get(w, 0) + 1
#         scored = []
#         for s in sentences:
#             score = sum(freq.get(w, 0) for w in re.findall(r"\w+", s.lower()))
#             scored.append((score, s))
#         scored.sort(reverse=True, key=lambda x: x[0])
#         out = []
#         cur = 0
#         for score, s in scored:
#             if cur + len(s) <= max_chars:
#                 out.append(s)
#                 cur += len(s)
#             if cur >= max_chars:
#                 break
#         if not out:
#             return sentences[0][:max_chars] + ("..." if len(sentences[0]) > max_chars else "")
#         summary = " ".join(out)
#         return summary if len(summary) <= max_chars else summary[:max_chars] + "..."

#     def get_status(self):
#         return {
#             "model": self.model_name,
#             "local_loaded": bool(self.local_pipe),
#             "hf_api_key_present": bool(self.hf_api_key),
#             "hf_endpoint": self.hf_endpoint
#         }
