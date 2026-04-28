"""v0: correctness probe for frozen-base disposition adapter branch.

Does base per-token entropy / top-2 margin / last hidden state carry signal to
distinguish correct from wrong responses on the Gemma 4 E2B Chef baseline?

Gate: AUC > 0.65 -> v1. AUC <= 0.65 -> frozen-base branch dead.
"""
import json, os, sys, time, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

MODEL = "unsloth/gemma-4-E2B-it"
DATA  = "/workspace/persistent/dd-v2/eval/results/baseline_gemma4_v2_from_step16.json"
OUT   = "/workspace/persistent/dd-v2/eval/results/uncert_probe_v0.json"

def build_inputs(tok, prompt, response):
    msgs = [{"role":"user","content":prompt}]
    p_out = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    p_ids = p_out["input_ids"] if hasattr(p_out, "input_ids") or isinstance(p_out, dict) else p_out
    r_ids = tok(response, return_tensors="pt", add_special_tokens=False).input_ids
    full = torch.cat([p_ids, r_ids], dim=1)
    return full, p_ids.shape[1], r_ids.shape[1]

@torch.no_grad()
def featurize(model, tok, prompt, response, device):
    full, plen, rlen = build_inputs(tok, prompt, response)
    full = full.to(device)
    if rlen < 2: return None
    # cap response to 512 tokens for speed
    if rlen > 512:
        full = full[:, :plen+512]; rlen = 512
    out = model(full, output_hidden_states=True)
    logits = out.logits[0]                # [T, V]
    hs = out.hidden_states[-1][0]         # [T, H]
    # predicting token t uses logits[t-1]; response tokens are at positions plen..plen+rlen-1
    pred_logits = logits[plen-1:plen+rlen-1]        # [rlen, V]
    logp = F.log_softmax(pred_logits.float(), dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(-1)                        # [rlen]
    top2 = pred_logits.float().topk(2, dim=-1).values
    margin = (top2[:,0] - top2[:,1])                 # [rlen]
    tgt = full[0, plen:plen+rlen]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [rlen]
    nll = -tok_logp
    hs_resp = hs[plen:plen+rlen].float()             # [rlen, H]
    def stats(x):
        x = x.cpu().numpy()
        return [float(x.mean()), float(x.std()), float(np.median(x)),
                float(np.quantile(x,0.9)), float(np.quantile(x,0.1)),
                float(x.max()), float(x.min())]
    feats = stats(ent) + stats(margin) + stats(nll)
    feats.append(float(rlen))
    hs_mean = hs_resp.mean(0).cpu().numpy()          # [H]
    hs_last = hs_resp[-1].cpu().numpy()              # [H]
    return np.array(feats, dtype=np.float32), hs_mean.astype(np.float32), hs_last.astype(np.float32)

def main():
    t0 = time.time()
    data = json.load(open(DATA))["items"]
    print(f"[v0] loaded {len(data)} items", flush=True)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    print(f"[v0] model loaded in {time.time()-t0:.1f}s", flush=True)

    X_scalar, X_hs_mean, X_hs_last, y = [], [], [], []
    for i, it in enumerate(data):
        try:
            r = featurize(model, tok, it["prompt"], it["response"], device)
        except Exception as e:
            print(f"  item {i} err: {e}", flush=True); continue
        if r is None: continue
        fs, hm, hl = r
        X_scalar.append(fs); X_hs_mean.append(hm); X_hs_last.append(hl)
        y.append(1 if it["correct"] else 0)
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(data)}] elapsed {time.time()-t0:.0f}s", flush=True)

    X_scalar = np.stack(X_scalar); X_hs_mean = np.stack(X_hs_mean); X_hs_last = np.stack(X_hs_last)
    y = np.array(y)
    print(f"[v0] features: scalar={X_scalar.shape} hs_mean={X_hs_mean.shape} y+={y.sum()} y-={len(y)-y.sum()}", flush=True)

    def cv_auc(X, name):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        aucs = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:,1]
            aucs.append(roc_auc_score(y[te], p))
        m, s = float(np.mean(aucs)), float(np.std(aucs))
        print(f"  {name}: AUC={m:.3f} ± {s:.3f}  folds={[round(a,3) for a in aucs]}", flush=True)
        return {"mean": m, "std": s, "folds": aucs}

    results = {
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y)-y.sum()),
        "scalar_dim": int(X_scalar.shape[1]), "hs_dim": int(X_hs_mean.shape[1]),
        "auc_scalar":      cv_auc(X_scalar, "scalar (21-d)"),
        "auc_hs_mean":     cv_auc(X_hs_mean, "hs_mean"),
        "auc_hs_last":     cv_auc(X_hs_last, "hs_last"),
        "auc_scalar+hsm":  cv_auc(np.concatenate([X_scalar, X_hs_mean], 1), "scalar+hs_mean"),
        "elapsed_sec": time.time()-t0,
    }
    best = max(results[k]["mean"] for k in results if k.startswith("auc_"))
    results["best_auc"] = best
    results["gate_pass_0.65"] = bool(best > 0.65)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT,"w"), indent=2)
    print(f"[v0] DONE best AUC={best:.3f} gate>{0.65}: {results['gate_pass_0.65']}  -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
