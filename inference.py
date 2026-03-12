"""
JurisGPT — Inference Script
Load model from HuggingFace and run legal Q&A
Last updated: 2026-03-12 05:06
"""
import torch, faiss, json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

HF_REPO = "Premchan369/JurisGPT"

def load_jurisgpt():
    print("Loading JurisGPT from HuggingFace...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    model     = AutoModelForCausalLM.from_pretrained(
        HF_REPO, quantization_config=bnb, device_map="auto"
    )
    embed     = SentenceTransformer("BAAI/bge-large-en")
    index     = faiss.read_index(hf_hub_download(HF_REPO, "jurisgpt_faiss.index"))
    with open(hf_hub_download(HF_REPO, "jurisgpt_metadata.json")) as f:
        meta = json.load(f)
    print("✅ JurisGPT loaded")
    return tokenizer, model, embed, index, meta


def ask(question, tokenizer, model, embed_model, index, meta, top_k=5):
    # Retrieve relevant legal context
    qv = embed_model.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, top_k)
    context = ""
    for score, i in zip(scores[0], ids[0]):
        if i < len(meta["all_docs"]):
            src  = meta["doc_sources"][i]
            text = meta["all_docs"][i]
            context += f"[{src}] (score={score:.3f})\n{text}\n\n"

    # Build prompt
    prompt = f"""Legal Query: {question}

Retrieved Legal Context:
{context[:2000]}

Provide detailed legal analysis with applicable laws, reasoning, and practical advice:"""

    messages = [
        {"role": "system", "content": "You are JurisGPT, expert Indian legal AI. Cite IPC sections and Constitutional articles."},
        {"role": "user",   "content": prompt}
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


if __name__ == "__main__":
    tokenizer, model, embed_model, index, meta = load_jurisgpt()
    print("JurisGPT ready. Type your legal question.")
    while True:
        q = input("\nQuestion (or \'quit\'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print("\n" + "="*50)
            print(ask(q, tokenizer, model, embed_model, index, meta))
            print("="*50)
