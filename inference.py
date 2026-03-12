"""JurisGPT v3 — Inference | 2026-03-12 09:20"""
import torch, faiss, json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

HF_REPO = "Premchan369/JurisGPT"

def load():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)
    tok  = AutoTokenizer.from_pretrained(HF_REPO)
    mdl  = AutoModelForCausalLM.from_pretrained(HF_REPO, quantization_config=bnb, device_map="auto")
    emb  = SentenceTransformer("BAAI/bge-large-en")
    idx  = faiss.read_index(hf_hub_download(HF_REPO, "jurisgpt_faiss.index"))
    with open(hf_hub_download(HF_REPO, "jurisgpt_metadata.json")) as f:
        meta = json.load(f)
    print(f"✅ JurisGPT v3 loaded — {meta['total_docs']:,} docs")
    return tok, mdl, emb, idx, meta

def ask(q, tok, mdl, emb, idx, meta):
    qv = emb.encode([q], normalize_embeddings=True).astype("float32")
    sc, ids = idx.search(qv, 5)
    ctx = "\n".join(f"[{meta['doc_sources'][i]}] {meta['all_docs'][i]}"
                     for s, i in zip(sc[0], ids[0]) if i < len(meta["all_docs"]))
    prompt = f"Legal Query: {q}\n\nLaws:\n{ctx[:2000]}\n\nAnalysis:"
    msgs = [{"role":"system","content":"You are JurisGPT, expert Indian legal AI."},
            {"role":"user","content":prompt}]
    t   = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(t, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=600, temperature=0.3,
                           do_sample=True, repetition_penalty=1.1,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

if __name__ == "__main__":
    tok, mdl, emb, idx, meta = load()
    while True:
        q = input("\nQuestion (quit to exit): ").strip()
        if q.lower() in ("quit","exit","q"): break
        if q: print("\n" + ask(q, tok, mdl, emb, idx, meta))
