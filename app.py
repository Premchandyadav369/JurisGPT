"""
JurisGPT Gradio Web App
Run locally: python app.py
Last updated: 2026-03-12 05:06
"""
import gradio as gr
from inference import load_jurisgpt, ask

print("Loading JurisGPT...")
tokenizer, model, embed_model, index, meta = load_jurisgpt()
print("✅ Ready")

def chat(message, history):
    return ask(message, tokenizer, model, embed_model, index, meta)

with gr.Blocks(title="⚖️ JurisGPT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚖️ JurisGPT — Indian Legal AI\n> For informational purposes only.")
    gr.ChatInterface(
        fn=chat,
        examples=[
            "Explain Article 21 of the Indian Constitution.",
            "What is IPC Section 302 — punishment for murder?",
            "What are my rights if arrested by police in India?",
            "My landlord is not returning security deposit. What can I do?",
            "Explain the difference between bail and anticipatory bail."
        ]
    )

if __name__ == "__main__":
    demo.launch(share=True)
