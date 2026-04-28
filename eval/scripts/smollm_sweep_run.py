"""SmolLM2-1.7B §11.6 replication: step16 feats + fresh gen/feats for disposition_probe_sweep."""
import os, sys, json, time
os.environ.setdefault("HF_HOME","/workspace/persistent/hf_cache")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RES = "/workspace/persistent/dd-v2/eval/results"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
BATCH = 8
MAXNEW = 512

def render(tok, prompt):
    return tok.apply_chat_template([{"role":"user","content":prompt}],
                                    add_generation_prompt=True, tokenize=False)

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    m.eval()
    return m, tok

def extract_hs_last(m, tok, rendered_prompts, responses):
    """Forward pass rendered_prompt+response, grab hs at last non-pad token."""
    X = []
    for bi in range(0, len(rendered_prompts), BATCH):
        rp = rendered_prompts[bi:bi+BATCH]
        rr = responses[bi:bi+BATCH]
        full = [p + r for p,r in zip(rp, rr)]
        enc = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False)
        ids = enc["input_ids"].to("cuda"); am = enc["attention_mask"].to("cuda")
        with torch.no_grad():
            out = m(ids, attention_mask=am, output_hidden_states=True)
        hs = out.hidden_states[-1]  # [B,T,H]
        for j in range(ids.shape[0]):
            # last non-pad pos for left-padded seq = T-1 always (padding is on left)
            pos = ids.shape[1] - 1
            X.append(hs[j, pos].float().cpu().numpy().astype(np.float32))
        print(f"  feat {min(bi+BATCH,len(rendered_prompts))}/{len(rendered_prompts)}", flush=True)
    return np.stack(X)

def generate_fresh(m, tok, prompts):
    rendered = [render(tok,p) for p in prompts]
    rows, resp_texts = [], []
    pad = tok.pad_token_id
    t0 = time.time()
    for bi in range(0, len(prompts), BATCH):
        batch = rendered[bi:bi+BATCH]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False)
        ids = enc["input_ids"].to("cuda"); am = enc["attention_mask"].to("cuda")
        plen = ids.shape[1]
        with torch.no_grad():
            out = m.generate(ids, attention_mask=am, max_new_tokens=MAXNEW,
                             do_sample=False, use_cache=True, pad_token_id=pad)
        for j in range(out.shape[0]):
            resp_ids = out[j, plen:]
            end = int((resp_ids != pad).sum().item())
            if end < 2:
                resp_texts.append(""); rows.append({"i":bi+j,"prompt":prompts[bi+j],"response":""}); continue
            text = tok.decode(resp_ids[:end], skip_special_tokens=True)
            resp_texts.append(text)
            rows.append({"i":bi+j,"prompt":prompts[bi+j],"response":text})
        print(f"  gen {min(bi+BATCH,len(prompts))}/{len(prompts)} t={time.time()-t0:.0f}s", flush=True)
    return rows, rendered, resp_texts

def main():
    m, tok = load_model()

    # --- step16 features ---
    eb = json.load(open(f"{RES}/eval_baseline.json"))["rows"]
    sm = json.load(open(f"{RES}/baseline_smollm_v2_samples.json"))["items"]
    assert len(eb) == len(sm) == 100
    s16_prompts_rendered = [render(tok, r["prompt"]) for r in eb]
    s16_responses = [it["samples"][0]["text"] for it in sm]
    s16_cov = [it["samples"][0]["coverage"] for it in sm]
    print(f"[step16] extracting features n={len(eb)}", flush=True)
    X16 = extract_hs_last(m, tok, s16_prompts_rendered, s16_responses)
    np.savez(f"{RES}/smollm_step16_features.npz", X=X16, coverage=np.array(s16_cov,dtype=np.float32))
    print(f"[step16] saved {X16.shape}", flush=True)

    # --- fresh generation + features ---
    fp = json.load(open(f"{RES}/v1_full_fresh_prompts.json"))["prompts"]
    print(f"[fresh] generating n={len(fp)}", flush=True)
    rows, rendered_fresh, resp_texts = generate_fresh(m, tok, fp)
    json.dump({"model":MODEL,"rows":rows}, open(f"{RES}/smollm_fresh_responses.json","w"), indent=2)
    print(f"[fresh] extracting features", flush=True)
    Xf = extract_hs_last(m, tok, rendered_fresh, resp_texts)
    np.savez(f"{RES}/smollm_fresh_features.npz", X=Xf)
    print(f"[fresh] saved responses={len(rows)} feats={Xf.shape}", flush=True)

if __name__ == "__main__":
    main()
