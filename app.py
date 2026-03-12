"""JurisGPT v3 Gradio App | 2026-03-12 09:20"""
import gradio as gr
from inference import load, ask

tok, mdl, emb, idx, meta = load()
demo = gr.ChatInterface(
    fn=lambda msg, hist: ask(msg, tok, mdl, emb, idx, meta),
    title="⚖️ JurisGPT v3 — Indian Legal AI",
    examples=["Explain Article 21","IPC Section 302?","Rights if arrested in India?"]
)
if __name__ == "__main__":
    demo.launch(share=True)
