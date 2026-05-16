import html
import os

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

from rag_engine import get_rag_engine

load_dotenv()

SYSTEM_PROMPT = """
Kamu adalah asisten AI untuk pencarian coffee shop berdasarkan data Google Maps.
Jawab selalu dalam Bahasa Indonesia.
Gunakan hanya context yang diberikan.
Jika informasi tidak ada di context, katakan dengan jujur bahwa data tidak tersedia.
Utamakan jawaban yang spesifik, faktual, dan menyebut nama tempat bila relevan.
""".strip()


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise gr.Error(
            "GROQ_API_KEY belum tersedia. Untuk Hugging Face Spaces, isi di Settings > Secrets."
        )
    return Groq(api_key=api_key)


def build_messages(history: list[tuple[str, str | None]]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_text, assistant_text in history:
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
    return messages


def build_rag_prompt(message: str, city: str, min_rating: float, open_now: bool):
    rag = get_rag_engine()
    retrieved_docs = rag.retrieve(
        message,
        k=5,
        city=city,
        min_rating=min_rating,
        open_now=open_now,
    )
    if not retrieved_docs:
        return None, []

    def format_open_label(value):
        if value is True:
            return "Ya"
        if value is False:
            return "Tidak"
        return "Tidak diketahui"

    context_sections = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        title = doc.metadata.get("title") or f"Dokumen {idx}"
        result_city = doc.metadata.get("city") or "Kota tidak tersedia"
        score = doc.metadata.get("score")
        reviews_count = doc.metadata.get("reviews_count")
        open_label = format_open_label(doc.metadata.get("open_now"))
        header = (
            f"[{idx}] {title} | kota: {result_city} | "
            f"rating: {score if score is not None else 'N/A'} | "
            f"reviews: {reviews_count if reviews_count is not None else 'N/A'} | "
            f"buka sekarang: {open_label}"
        )
        context_sections.append(f"{header}\n{doc.text}")

    filters_summary = (
        f"Filter aktif: kota={city}, rating minimum={min_rating:.1f}, "
        f"buka sekarang={'ya' if open_now else 'tidak'}."
    )
    context = "\n\n".join(context_sections)
    prompt = f"""
Jawab pertanyaan pengguna berdasarkan context berikut.
Fokus pada data coffee shop dari Google Maps.

{filters_summary}

Context:
{context}

Pertanyaan:
{message}

Instruksi jawaban:
- Jawab ringkas tapi informatif.
- Jika pengguna meminta rekomendasi, bandingkan beberapa tempat dari context.
- Jika pengguna bertanya lokasi, jam buka, rating, review, atau tingkat keramaian, ambil dari context.
- Jika data tidak cukup, katakan bahwa data pada dataset belum memuat informasi itu.
""".strip()
    return prompt, retrieved_docs


def render_sources(retrieved_docs: list) -> str:
    if not retrieved_docs:
        return "<p>Tidak ada source result yang cocok dengan filter saat ini.</p>"

    def format_open_label(value):
        if value is True:
            return "Buka sekarang"
        if value is False:
            return "Sedang tutup"
        return "Status buka tidak diketahui"

    blocks = ["<div>"]
    for idx, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata
        title = html.escape(meta.get("title") or f"Hasil {idx}")
        city = html.escape(meta.get("city") or "Kota tidak tersedia")
        address = html.escape(meta.get("address") or "Alamat tidak tersedia")
        opening_hours = html.escape(meta.get("opening_hours") or "Jam buka tidak tersedia")
        rating = meta.get("score")
        reviews = meta.get("reviews_count")
        phone = html.escape(meta.get("phone") or "Tidak tersedia")
        busy = html.escape(meta.get("popular_live_text") or "Tidak tersedia")
        open_label = format_open_label(meta.get("open_now"))
        maps_url = meta.get("maps_url")

        lines = [
            f"<div style='padding:12px; margin-bottom:12px; border:1px solid #d7d7d7; border-radius:10px;'>",
            f"<div><strong>#{idx}. {title}</strong></div>",
            f"<div>Kota: {city}</div>",
            f"<div>Rating: {rating if rating is not None else 'N/A'} | Reviews: {reviews if reviews is not None else 'N/A'}</div>",
            f"<div>{open_label}</div>",
            f"<div>Alamat: {address}</div>",
            f"<div>Jam buka: {opening_hours}</div>",
            f"<div>Telepon: {phone}</div>",
            f"<div>Popular times: {busy}</div>",
        ]
        if maps_url:
            safe_url = html.escape(maps_url, quote=True)
            lines.append(
                f"<div><a href='{safe_url}' target='_blank' rel='noopener noreferrer'>Buka di Google Maps</a></div>"
            )
        lines.append("</div>")
        blocks.append("".join(lines))
    blocks.append("</div>")
    return "".join(blocks)


def ask_assistant(message: str, chat_history: list, city: str, min_rating: float, open_now: bool):
    history = chat_history or []
    clean_message = (message or "").strip()
    if not clean_message:
        return history, history, "", "<p>Tulis pertanyaan terlebih dahulu.</p>"

    prompt, retrieved_docs = build_rag_prompt(clean_message, city, min_rating, open_now)
    if not retrieved_docs:
        answer = (
            "Saya belum menemukan coffee shop yang cocok dengan filter dan pertanyaan Anda. "
            "Coba longgarkan filter kota, turunkan rating minimum, atau matikan opsi buka sekarang."
        )
        updated_history = history + [(clean_message, answer)]
        return updated_history, updated_history, "", render_sources([])

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    client = get_client()
    messages = build_messages(history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    answer = response.choices[0].message.content or "Maaf, saya belum bisa menjawab."
    updated_history = history + [(clean_message, answer)]
    return updated_history, updated_history, "", render_sources(retrieved_docs)


def clear_chat():
    return [], [], "", "<p>Source results akan muncul di sini setelah Anda bertanya.</p>"


rag = get_rag_engine()
city_choices = ["Semua kota"] + rag.cities

with gr.Blocks(title="Coffee Shop RAG Chatbot") as demo:
    gr.Markdown(
        """
        # Coffee Shop RAG Chatbot
        Tanya coffee shop dari dataset Google Maps, lengkap dengan filter dan source results.
        """
    )

    with gr.Row():
        city_input = gr.Dropdown(
            choices=city_choices,
            value="Semua kota",
            label="Filter Kota",
        )
        rating_input = gr.Slider(
            minimum=0,
            maximum=5,
            value=0,
            step=0.1,
            label="Rating Minimum",
        )
        open_now_input = gr.Checkbox(
            value=False,
            label="Buka Sekarang",
        )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Percakapan", height=500)
            message_input = gr.Textbox(
                placeholder="Contoh: coffee shop terbaik di Bandar Lampung",
                label="Pertanyaan",
            )
            with gr.Row():
                send_button = gr.Button("Kirim", variant="primary")
                clear_button = gr.Button("Clear")
        with gr.Column(scale=2):
            gr.Markdown("### Source Results")
            sources_output = gr.HTML("<p>Source results akan muncul di sini setelah Anda bertanya.</p>")

    history_state = gr.State([])

    send_button.click(
        ask_assistant,
        inputs=[message_input, history_state, city_input, rating_input, open_now_input],
        outputs=[chatbot, history_state, message_input, sources_output],
    )
    message_input.submit(
        ask_assistant,
        inputs=[message_input, history_state, city_input, rating_input, open_now_input],
        outputs=[chatbot, history_state, message_input, sources_output],
    )
    clear_button.click(
        clear_chat,
        outputs=[chatbot, history_state, message_input, sources_output],
    )


if __name__ == "__main__":
    demo.launch()
