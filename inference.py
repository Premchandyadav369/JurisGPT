#!/usr/bin/env python3
"""
JurisGPT — Indian Legal AI
Load and run inference from Hugging Face: Premchan369/JurisGPT
"""

import torch
import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

HF_REPO = "Premchan369/JurisGPT"

def load_jurisgpt():
    print("Loading JurisGPT...")
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

    # Load FAISS index
    faiss_path = hf_hub_download(HF_REPO, "jurisgpt_faiss.index")
    index      = faiss.read_index(faiss_path)

    # Load metadata
    meta_path  = hf_hub_download(HF_REPO, "jurisgpt_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    print("✅ JurisGPT loaded")
    return tokenizer, model, embed, index, meta


def ask(question, tokenizer, model, embed_model, index, meta, top_k=4):
    q_vec = embed_model.encode([question], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_vec, top_k)
    context = ""
    for s, i in zip(scores[0], idxs[0]):
        if i < len(meta["all_docs"]):
            context += f"[{meta['doc_sources'][i]}]\n{meta['all_docs'][i]}\n\n"

    prompt = f"Question: {question}\n\nRelevant Laws:\n{context[:1500]}\n\nProvide legal analysis:"
    messages = [
        {"role": "system", "content": "You are JurisGPT, an expert Indian legal AI."},
        {"role": "user",   "content": prompt}
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.3, do_sample=True)
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer, model, embed_model, index, meta = load_jurisgpt()
    while True:
        q = input("\nYour legal question (or \'quit\'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print("\n" + ask(q, tokenizer, model, embed_model, index, meta))
