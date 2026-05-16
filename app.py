import os

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SYSTEM_PROMPT = "Kamu adalah chatbot yang membantu dengan jawaban singkat, jelas, dan ramah."


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise gr.Error(
            "GROQ_API_KEY belum tersedia. Untuk Hugging Face Spaces, isi di Settings > Secrets."
        )
    return Groq(api_key=api_key)


def build_messages(message: str, history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})
    return messages


def chat(message: str, history: list[dict]) -> str:
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    client = get_client()
    messages = build_messages(message, history)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content or "Maaf, saya belum bisa menjawab."


with gr.Blocks(title="Simple Groq Chatbot") as demo:
    gr.Markdown(
        """
        # Simple Groq Chatbot
        Chatbot LLM sederhana menggunakan Groq API.
        """
    )

    gr.ChatInterface(
        fn=chat,
        type="messages",
        textbox=gr.Textbox(
            placeholder="Tulis pertanyaan di sini...",
            container=False,
            scale=8,
        ),
        chatbot=gr.Chatbot(height=500),
        additional_inputs=[],
    )


if __name__ == "__main__":
    demo.launch()
