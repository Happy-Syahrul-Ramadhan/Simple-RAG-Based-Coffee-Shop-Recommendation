---
title: Simple Groq Chatbot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# Simple Groq Chatbot

Setup chatbot Groq berbasis RAG untuk tanya jawab coffee shop dari dataset Google Maps di `data/data-maps.json`.

Fitur utama:

1. Filter kota
2. Rating minimum
3. Opsi `buka sekarang`
4. Source results berurutan dari hasil retrieval
5. Link langsung ke Google Maps dari koordinat `lat,lng`

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

Saat pertama kali dijalankan, aplikasi akan:

1. Memfilter entri yang relevan dengan coffee shop
2. Membentuk dokumen konteks dari field seperti nama, kategori, alamat, rating, review, jam buka, popular times, menu, telepon, dan koordinat
3. Membuat embedding dengan `sentence-transformers`
4. Menyusun index FAISS untuk retrieval
5. Mengirim context hasil retrieval ke Groq untuk menjawab pertanyaan pengguna

## Deploy ke Hugging Face Spaces

1. Buat Space baru dengan SDK `Gradio`.
2. Upload file project ini ke Space.
3. Buka `Settings > Secrets`.
4. Tambahkan secret `GROQ_API_KEY`.
5. Opsional: tambahkan `GROQ_MODEL`.
6. Opsional: tambahkan `EMBEDDING_MODEL` jika ingin mengganti model embedding.

Contoh value:

```text
GROQ_MODEL=llama-3.1-8b-instant
```

Setelah build selesai, chatbot akan otomatis tampil sebagai web app.
