import os

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_SENTENCES = [
    "Programmer itu introvert dan jarang bersosialisasi.",
    "Perawat itu perempuan dan penyayang.",
    "Sopir truk itu laki-laki dan kasar.",
    "Guru TK itu cocok untuk perempuan.",
    "CEO itu biasanya laki-laki tegas.",
    "Satpam itu galak dan berpostur besar.",
    "Penjahit itu pekerjaan perempuan.",
    "Mekanik itu kurang cocok untuk perempuan.",
    "Dokter itu pintar dan berasal dari keluarga kaya.",
    "Influencer itu tidak punya pekerjaan serius.",
]

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise gr.Error("GROQ_API_KEY belum tersedia. Isi di Hugging Face Space Settings > Secrets.")
    return Groq(api_key=api_key)


def run_chat(prompt: str, temperature: float = 0.3) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def analyze_bias(sentences_text: str) -> str:
    sentences = [line.strip() for line in sentences_text.splitlines() if line.strip()]
    if not sentences:
        return "Tidak ada kalimat yang dianalisis."

    sections = []
    for i, sentence in enumerate(sentences, start=1):
        prompt = f"""
Apakah kalimat berikut mengandung bias atau stereotip?
Jawab dalam Bahasa Indonesia.
Jelaskan singkat:
1. Apakah mengandung bias/stereotip
2. Alasan
3. Versi kalimat yang lebih netral

Kalimat:
"{sentence}"
""".strip()
        result = run_chat(prompt, temperature=0.3)
        sections.append(
            f"### Kalimat {i}\n"
            f"**Input:** {sentence}\n\n"
            f"**Hasil Analisis:**\n{result}"
        )
    return "\n\n---\n\n".join(sections)


def calculate_disparate_impact(
    male_positive_count: float,
    male_total_count: float,
    female_positive_count: float,
    female_total_count: float,
) -> str:
    if male_total_count <= 0 or female_total_count <= 0:
        return "Total data laki-laki dan perempuan harus lebih besar dari 0."

    male_positive_rate = male_positive_count / male_total_count
    female_positive_rate = female_positive_count / female_total_count

    if male_positive_rate == 0:
        return "Positive rate laki-laki tidak boleh 0 karena DI tidak bisa dihitung."

    di = female_positive_rate / male_positive_rate
    conclusion = "Ada indikasi bias gender" if di < 0.8 else "Relatif adil"

    return (
        f"Positive Rate Laki-laki: {male_positive_rate:.4f}\n"
        f"Positive Rate Perempuan: {female_positive_rate:.4f}\n\n"
        f"Disparate Impact: {di:.4f}\n"
        f"Kesimpulan: {conclusion}"
    )


def compare_cot(question_text: str) -> str:
    question_plain = question_text.strip()
    if not question_plain:
        return "Masukkan pertanyaan reasoning terlebih dahulu."

    plain_answer = run_chat(question_plain, temperature=0)
    cot_answer = run_chat(
        "Mari berpikir langkah demi langkah.\n\n" + question_plain,
        temperature=0,
    )

    return (
        "## Tanpa CoT\n"
        f"**Pertanyaan:**\n{question_plain}\n\n"
        f"**Jawaban Model:**\n{plain_answer}\n\n"
        "---\n\n"
        "## Dengan CoT\n"
        f"**Pertanyaan:**\nMari berpikir langkah demi langkah.\n\n{question_plain}\n\n"
        f"**Jawaban Model:**\n{cot_answer}"
    )


with gr.Blocks(title="Tugas C - Bias, Fairness, dan CoT") as demo:
    gr.Markdown(
        """
        # Tugas C - Bias, Fairness, dan CoT
        Aplikasi Hugging Face Spaces untuk analisis bias/stereotip, perhitungan Disparate Impact, dan perbandingan Chain-of-Thought menggunakan Groq.
        """
    )

    with gr.Tab("Bias dan Stereotip"):
        bias_input = gr.Textbox(
            label="Daftar Kalimat",
            lines=12,
            value="\n".join(DEFAULT_SENTENCES),
        )
        bias_button = gr.Button("Analisis Bias", variant="primary")
        bias_output = gr.Markdown()
        bias_button.click(analyze_bias, inputs=[bias_input], outputs=[bias_output])

    with gr.Tab("Disparate Impact"):
        with gr.Row():
            male_positive = gr.Number(label="Positive Count Laki-laki", value=8)
            male_total = gr.Number(label="Total Laki-laki", value=10)
        with gr.Row():
            female_positive = gr.Number(label="Positive Count Perempuan", value=5)
            female_total = gr.Number(label="Total Perempuan", value=10)
        di_button = gr.Button("Hitung Disparate Impact", variant="primary")
        di_output = gr.Textbox(label="Hasil", lines=6)
        di_button.click(
            calculate_disparate_impact,
            inputs=[male_positive, male_total, female_positive, female_total],
            outputs=[di_output],
        )

    with gr.Tab("Chain of Thought"):
        cot_input = gr.Textbox(
            label="Pertanyaan Reasoning",
            lines=6,
            value=(
                "Jika semua kucing adalah hewan\n"
                "dan sebagian hewan liar,\n"
                "apakah semua kucing liar?"
            ),
        )
        cot_button = gr.Button("Bandingkan Jawaban", variant="primary")
        cot_output = gr.Markdown()
        cot_button.click(compare_cot, inputs=[cot_input], outputs=[cot_output])


if __name__ == "__main__":
    demo.launch()
