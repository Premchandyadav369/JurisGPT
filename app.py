"""
JurisGPT Gradio Web App
Run locally: python app.py
"""
import gradio as gr
from inference import load_jurisgpt, ask

tokenizer, model, embed_model, index, meta = load_jurisgpt()

def chat(message, history):
    return ask(message, tokenizer, model, embed_model, index, meta)

demo = gr.ChatInterface(
    fn=chat,
    title="⚖️ JurisGPT — Indian Legal AI",
    examples=[
        "Explain Article 21 of the Indian Constitution.",
        "What does IPC Section 302 say about murder?",
        "What are my rights if arrested by police in India?"
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
