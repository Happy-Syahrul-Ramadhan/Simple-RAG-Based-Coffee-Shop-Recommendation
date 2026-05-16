import html
import os
import re

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


def build_messages(history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def parse_float_or_none(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_city_filter(message: str, selected_city: str, available_cities: list[str]) -> str:
    if selected_city and selected_city != "Semua kota":
        return selected_city

    query_lower = (message or "").lower()
    city_aliases = {
        "metro": "Kota Metro",
        "kota metro": "Kota Metro",
        "bandar lampung": "Kota Bandar Lampung",
        "kota bandar lampung": "Kota Bandar Lampung",
        "lampung timur": "Kabupaten Lampung Timur",
        "kabupaten lampung timur": "Kabupaten Lampung Timur",
        "lampung selatan": "Kabupaten Lampung Selatan",
        "kabupaten lampung selatan": "Kabupaten Lampung Selatan",
        "lampung tengah": "Kabupaten Lampung Tengah",
        "kabupaten lampung tengah": "Kabupaten Lampung Tengah",
    }

    for alias, resolved_city in city_aliases.items():
        if alias in query_lower and resolved_city in available_cities:
            return resolved_city

    return "Semua kota"


def build_rag_prompt(
    message: str,
    city: str,
    min_rating: float,
    open_now: bool,
    user_lat: float | None,
    user_lng: float | None,
):
    rag = get_rag_engine()
    retrieved_docs = rag.retrieve(
        message,
        k=5,
        city=city,
        min_rating=min_rating,
        open_now=open_now,
        user_lat=user_lat,
        user_lng=user_lng,
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
        requested_hour = doc.metadata.get("requested_hour")
        occupancy = doc.metadata.get("requested_occupancy")
        occupancy_label = doc.metadata.get("requested_occupancy_label")
        distance_km = doc.metadata.get("distance_km")
        occupancy_text = "tidak tersedia"
        distance_text = "tidak tersedia"
        if requested_hour is not None and occupancy is not None:
            occupancy_text = f"{requested_hour:02d}.00 -> {occupancy_label} ({occupancy:.1f}%)"
        if distance_km is not None:
            distance_text = f"{distance_km:.2f} km"
        header = (
            f"[{idx}] {title} | kota: {result_city} | "
            f"rating: {score if score is not None else 'N/A'} | "
            f"reviews: {reviews_count if reviews_count is not None else 'N/A'} | "
            f"buka sekarang: {open_label} | "
            f"keramaian_jam_diminta: {occupancy_text} | "
            f"jarak_dari_pengguna: {distance_text}"
        )
        context_sections.append(f"{header}\n{doc.text}")

    filters_summary = (
        f"Filter aktif: kota={city}, rating minimum={min_rating:.1f}, "
        f"buka sekarang={'ya' if open_now else 'tidak'}."
    )
    location_summary = (
        f"Lokasi pengguna: lat={user_lat}, lng={user_lng}."
        if user_lat is not None and user_lng is not None
        else "Lokasi pengguna tidak diisi."
    )
    context = "\n\n".join(context_sections)
    prompt = f"""
Jawab pertanyaan pengguna berdasarkan context berikut.
Fokus pada data coffee shop dari Google Maps.

{filters_summary}
{location_summary}

Context:
{context}

Pertanyaan:
{message}

Instruksi jawaban:
- Jawab ringkas tapi informatif.
- Jika pengguna meminta rekomendasi, bandingkan beberapa tempat dari context.
- Jika pengguna bertanya lokasi, jam buka, rating, review, atau tingkat keramaian, ambil dari context.
- Jika data tidak cukup, katakan bahwa data pada dataset belum memuat informasi itu.
- Saat merekomendasikan tempat, gunakan format `[ID] Nama Tempat` agar sumber bisa ditelusuri.
- Untuk pertanyaan soal jam ramai atau sepi, prioritaskan `popularTimesHistogram`, bukan `popularTimesLiveText`.
- Untuk pertanyaan soal tempat terdekat, prioritaskan `jarak_dari_pengguna` jika tersedia.
- Jika pertanyaan meminta tempat terdekat, anggap urutan context sudah disusun dari jarak paling dekat ke yang lebih jauh.
""".strip()
    return prompt, retrieved_docs


def extract_recommended_docs(answer: str, retrieved_docs: list) -> list:
    if not answer or not retrieved_docs:
        return retrieved_docs

    answer_lower = answer.lower()
    selected = []
    selected_indexes = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}

    for idx, doc in enumerate(retrieved_docs, start=1):
        title = (doc.metadata.get("title") or "").strip().lower()
        if idx in selected_indexes or (title and title in answer_lower):
            selected.append(doc)

    return selected or retrieved_docs


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
        requested_hour = meta.get("requested_hour")
        requested_occupancy = meta.get("requested_occupancy")
        requested_label = html.escape(meta.get("requested_occupancy_label") or "Tidak tersedia")
        distance_km = meta.get("distance_km")

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
        if distance_km is not None:
            lines.append(f"<div>Jarak dari lokasi Anda: {distance_km:.2f} km</div>")
        if requested_hour is not None:
            if requested_occupancy is not None:
                lines.append(
                    f"<div>Keramaian sekitar pukul {requested_hour:02d}.00: {requested_label} ({requested_occupancy:.1f}%)</div>"
                )
            else:
                lines.append(
                    f"<div>Keramaian sekitar pukul {requested_hour:02d}.00: Tidak tersedia</div>"
                )
        if maps_url:
            safe_url = html.escape(maps_url, quote=True)
            lines.append(
                f"<div><a href='{safe_url}' target='_blank' rel='noopener noreferrer'>Buka di Google Maps</a></div>"
            )
        lines.append("</div>")
        blocks.append("".join(lines))
    blocks.append("</div>")
    return "".join(blocks)


def ask_assistant(
    message: str,
    chat_history: list,
    city: str,
    min_rating: float,
    open_now: bool,
    user_lat_input,
    user_lng_input,
):
    history = chat_history or []
    clean_message = (message or "").strip()
    if not clean_message:
        return history, history, "", "<p>Tulis pertanyaan terlebih dahulu.</p>"

    user_lat = parse_float_or_none(user_lat_input)
    user_lng = parse_float_or_none(user_lng_input)
    effective_city = resolve_city_filter(clean_message, city, city_choices[1:])
    prompt, retrieved_docs = build_rag_prompt(
        clean_message,
        effective_city,
        min_rating,
        open_now,
        user_lat,
        user_lng,
    )
    if not retrieved_docs:
        answer = (
            "Saya belum menemukan coffee shop yang cocok dengan filter dan pertanyaan Anda. "
            "Coba longgarkan filter kota, turunkan rating minimum, atau matikan opsi buka sekarang."
        )
        updated_history = history + [
            {"role": "user", "content": clean_message},
            {"role": "assistant", "content": answer},
        ]
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
    recommended_docs = extract_recommended_docs(answer, retrieved_docs)
    updated_history = history + [
        {"role": "user", "content": clean_message},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, updated_history, "", render_sources(recommended_docs)


def clear_chat():
    return [], [], "", "<p>Source results akan muncul di sini setelah Anda bertanya.</p>"


def keep_location_values(user_lat_input, user_lng_input, location_status):
    return user_lat_input, user_lng_input, location_status


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
        locate_button = gr.Button("Izinkan lokasi untuk pencarian terdekat")
        location_status = gr.Textbox(
            value="Lokasi belum diaktifkan.",
            label="Status Lokasi",
            interactive=False,
        )

    user_lat_input = gr.Textbox(value="", visible=False)
    user_lng_input = gr.Textbox(value="", visible=False)

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

    locate_button.click(
        keep_location_values,
        inputs=[user_lat_input, user_lng_input, location_status],
        outputs=[user_lat_input, user_lng_input, location_status],
        js="""
        async (currentLat, currentLng, currentStatus) => {
          const save = (lat, lng, status) => {
            try {
              localStorage.setItem("coffee_rag_user_lat", String(lat));
              localStorage.setItem("coffee_rag_user_lng", String(lng));
            } catch (e) {}
            return [String(lat), String(lng), status];
          };

          if (!navigator.geolocation) {
            return [currentLat || "", currentLng || "", "Browser ini tidak mendukung geolocation."];
          }

          return await new Promise((resolve) => {
            navigator.geolocation.getCurrentPosition(
              (position) => {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                resolve(save(lat, lng, `Lokasi aktif: ${lat.toFixed(6)}, ${lng.toFixed(6)}`));
              },
              (error) => {
                const message = error && error.message ? error.message : "Gagal mengambil lokasi.";
                resolve([currentLat || "", currentLng || "", `Lokasi gagal diambil: ${message}`]);
              },
              { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 }
            );
          });
        }
        """,
    )

    demo.load(
        keep_location_values,
        inputs=[user_lat_input, user_lng_input, location_status],
        outputs=[user_lat_input, user_lng_input, location_status],
        js="""
        async (currentLat, currentLng, currentStatus) => {
          const fallback = () => {
            try {
              const lat = localStorage.getItem("coffee_rag_user_lat") || currentLat || "";
              const lng = localStorage.getItem("coffee_rag_user_lng") || currentLng || "";
              const status = lat && lng
                ? `Menggunakan lokasi tersimpan terakhir: ${Number(lat).toFixed(6)}, ${Number(lng).toFixed(6)}`
                : (currentStatus || "Lokasi belum diaktifkan.");
              return [lat, lng, status];
            } catch (e) {
              return [currentLat || "", currentLng || "", currentStatus || "Lokasi belum diaktifkan."];
            }
          };

          if (!navigator.geolocation) {
            return fallback();
          }

          return await new Promise((resolve) => {
            navigator.geolocation.getCurrentPosition(
              (position) => {
                const lat = String(position.coords.latitude);
                const lng = String(position.coords.longitude);
                try {
                  localStorage.setItem("coffee_rag_user_lat", lat);
                  localStorage.setItem("coffee_rag_user_lng", lng);
                } catch (e) {}
                resolve([lat, lng, `Lokasi diperbarui otomatis: ${Number(lat).toFixed(6)}, ${Number(lng).toFixed(6)}`]);
              },
              () => {
                resolve(fallback());
              },
              { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
            );
          });
        }
        """,
    )

    send_button.click(
        ask_assistant,
        inputs=[
            message_input,
            history_state,
            city_input,
            rating_input,
            open_now_input,
            user_lat_input,
            user_lng_input,
        ],
        outputs=[chatbot, history_state, message_input, sources_output],
    )
    message_input.submit(
        ask_assistant,
        inputs=[
            message_input,
            history_state,
            city_input,
            rating_input,
            open_now_input,
            user_lat_input,
            user_lng_input,
        ],
        outputs=[chatbot, history_state, message_input, sources_output],
    )
    clear_button.click(
        clear_chat,
        outputs=[chatbot, history_state, message_input, sources_output],
    )


if __name__ == "__main__":
    demo.launch()
