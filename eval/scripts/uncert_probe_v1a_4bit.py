"""v0 + v1a under Unsloth FastModel load_in_4bit=True to match step16 baseline
generation config exactly (same model state, same dtype, same quant).
"""
import json, os, time, numpy as np, torch
import torch.nn.functional as F
from unsloth import FastModel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

MODEL = "unsloth/gemma-4-E2B-it"
DATA  = "/workspace/persistent/dd-v2/eval/results/baseline_gemma4_v2_from_step16.json"
OUT   = "/workspace/persistent/dd-v2/eval/results/uncert_probe_v1a_4bit.json"

def build_inputs(tok, text_tok, prompt, response):
    msgs = [{"role":"user","content":[{"type":"text","text":prompt}]}]
    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p_ids = text_tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids
    r_ids = text_tok(response, return_tensors="pt", add_special_tokens=False).input_ids
    return torch.cat([p_ids, r_ids], dim=1), p_ids.shape[1], r_ids.shape[1]

@torch.no_grad()
def featurize(model, tok, text_tok, prompt, response, device):
    full, plen, rlen = build_inputs(tok, text_tok, prompt, response)
    full = full.to(device)
    if rlen < 2: return None
    if rlen > 512:
        full = full[:, :plen+512]; rlen = 512
    out = model(full, output_hidden_states=True)
    logits = out.logits[0]
    hs = out.hidden_states[-1][0]
    pred_logits = logits[plen-1:plen+rlen-1]
    logp = F.log_softmax(pred_logits.float(), dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(-1)
    top2 = pred_logits.float().topk(2, dim=-1).values
    margin = (top2[:,0] - top2[:,1])
    tgt = full[0, plen:plen+rlen]
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    hs_resp = hs[plen:plen+rlen].float()
    def stats(x):
        x = x.cpu().numpy()
        return [float(x.mean()), float(x.std()), float(np.median(x)),
                float(np.quantile(x,0.9)), float(np.quantile(x,0.1)),
                float(x.max()), float(x.min())]
    feats = stats(ent) + stats(margin) + stats(nll) + [float(rlen)]
    hs_mean = hs_resp.mean(0).cpu().numpy()
    hs_last = hs_resp[-1].cpu().numpy()
    return (np.array(feats, dtype=np.float32),
            hs_mean.astype(np.float32), hs_last.astype(np.float32))

def main():
    t0 = time.time()
    data = json.load(open(DATA))["items"]
    model, tok = FastModel.from_pretrained(MODEL, max_seq_length=2048,
                                           load_in_4bit=True, full_finetuning=False)
    FastModel.for_inference(model)
    model.eval()
    text_tok = tok.tokenizer if hasattr(tok, "tokenizer") else tok
    print(f"[v1a4bit] model loaded 4-bit in {time.time()-t0:.1f}s", flush=True)

    X_scalar, X_mean, X_last, y = [], [], [], []
    for i, it in enumerate(data):
        try:
            r = featurize(model, tok, text_tok, it["prompt"], it["response"], "cuda")
        except Exception as e:
            print(f"  item {i} err: {e}", flush=True); continue
        if r is None: continue
        X_scalar.append(r[0]); X_mean.append(r[1]); X_last.append(r[2])
        y.append(1 if it["correct"] else 0)
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(data)}] t={time.time()-t0:.0f}s", flush=True)
    X_scalar = np.stack(X_scalar); X_mean = np.stack(X_mean); X_last = np.stack(X_last)
    y = np.array(y)
    print(f"[v1a4bit] features n={len(y)} pos={int(y.sum())}", flush=True)

    del model; torch.cuda.empty_cache()

    def oof(clf_factory, X):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        p = np.zeros(len(y), dtype=np.float32)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = clf_factory(); clf.fit(sc.transform(X[tr]), y[tr])
            p[te] = clf.predict_proba(sc.transform(X[te]))[:,1]
        return p, float(roc_auc_score(y, p))

    def gate(p_correct):
        out = {}
        n_w = int((y==0).sum()); n_c = int((y==1).sum())
        best = None
        for thr in np.linspace(0.05, 0.95, 91):
            f = (p_correct > thr).astype(int)
            p_w = float(f[y==0].mean()); p_c = float(f[y==1].mean())
            asym = p_w - p_c
            if best is None or asym < best["asym"]:
                best = {"thr":float(thr),"p_assert_wrong":p_w,"p_assert_correct":p_c,"asym":asym}
        out["min"] = best
        bal = None
        for thr in np.linspace(0.05, 0.95, 91):
            f = (p_correct > thr).astype(int)
            p_c = float(f[y==1].mean()); p_w = float(f[y==0].mean())
            if p_c >= 0.70:
                asym = p_w - p_c
                if bal is None or asym < bal["asym"]:
                    bal = {"thr":float(thr),"p_assert_wrong":p_w,"p_assert_correct":p_c,"asym":asym}
        out["balanced_pc_0.70"] = bal
        return out

    results = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y)-y.sum()),
               "load": "unsloth FastModel load_in_4bit=True",
               "baseline_assertion_asym": -0.009, "bar_assertion_asym": -0.15}
    for name, X in [("scalar", X_scalar), ("hs_mean", X_mean), ("hs_last", X_last)]:
        for clf_name, factory in [
            ("logreg",     lambda: LogisticRegression(max_iter=2000, C=1.0)),
            ("logreg_C0.1",lambda: LogisticRegression(max_iter=2000, C=0.1)),
            ("mlp_32",     lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=0)),
        ]:
            p, auc = oof(factory, X)
            g = gate(p)
            bal = g["balanced_pc_0.70"]
            bal_str = "none" if bal is None else f"{bal['asym']:.3f}@{bal['thr']:.2f}"
            print(f"  {name}+{clf_name}: AUC={auc:.3f}  min={g['min']['asym']:.3f}  bal={bal_str}", flush=True)
            results[f"{name}__{clf_name}"] = {"auc": auc, "gate": g}

    # best balanced across all configs
    bb = None
    for k, v in results.items():
        if not isinstance(v, dict) or "gate" not in v: continue
        bal = v["gate"].get("balanced_pc_0.70")
        if bal is None: continue
        if bb is None or bal["asym"] < bb[1]["asym"]:
            bb = (k, bal)
    results["best_balanced"] = {"config": bb[0], **bb[1]} if bb else None
    results["gate_pass"] = bool(bb and bb[1]["asym"] <= -0.15)
    print(f"[v1a4bit] BEST BALANCED: {bb[0] if bb else 'none'} asym={bb[1]['asym'] if bb else 'na'}", flush=True)
    print(f"[v1a4bit] gate_pass={results['gate_pass']}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT,"w"), indent=2, default=float)
    print(f"[v1a4bit] DONE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
