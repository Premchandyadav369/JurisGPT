"""JurisGPT Gradio App | 2026-03-12 16:57"""
import gradio as gr
from inference import load, ask

tok, mdl, emb, idx, meta = load()
demo = gr.ChatInterface(
    fn=lambda msg, hist: ask(msg, tok, mdl, emb, idx, meta),
    title="JurisGPT — Indian Legal AI",
    examples=[
        "Explain Article 21",
        "IPC Section 302?",
        "Rights if arrested in India?",
        "My employer has not paid salary — what can I do?",
    ]
)
if __name__ == "__main__":
    demo.launch(share=True)
