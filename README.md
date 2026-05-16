---
title: Simple Groq Chatbot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
---

# Simple Groq Chatbot

Setup sederhana chatbot LLM berbasis Groq yang siap dipakai di Hugging Face Spaces.

## Jalankan lokal

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Isi `GROQ_API_KEY` di file `.env`, lalu jalankan:

```powershell
python app.py
```

## Deploy ke Hugging Face Spaces

1. Buat Space baru dengan SDK `Gradio`.
2. Upload file project ini ke Space.
3. Buka `Settings > Secrets`.
4. Tambahkan secret `GROQ_API_KEY`.
5. Opsional: tambahkan `GROQ_MODEL` jika ingin ganti model.

Contoh value:

```text
GROQ_MODEL=llama-3.1-8b-instant
```

Setelah build selesai, chatbot akan otomatis tampil sebagai web app.
