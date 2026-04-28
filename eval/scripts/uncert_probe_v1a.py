"""v1a: supervised gate diagnostic on existing n=100.

Reuse v0 feature extraction. For each CV fold, train a gate on train split,
predict on test split, and compute post-gate assertion asymmetry:
  - item predicted "wrong" -> hedged (assert=0)
  - item predicted "correct" -> asserted (assert=1)

Gate: post-gate assertion_asymmetry = p_assert_wrong - p_assert_correct <= -0.15
Coverage is unchanged (hedging is prefix-only, content identical), so
coverage >= 0.43 is trivially met at baseline 0.58.

If this clears the bar on out-of-fold predictions, scale to v1-full (1k items,
API judges). If not, supervised ceiling is below the bar, stop.
"""
import json, os, time, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

MODEL = "unsloth/gemma-4-E2B-it"
DATA  = "/workspace/persistent/dd-v2/eval/results/baseline_gemma4_v2_from_step16.json"
OUT   = "/workspace/persistent/dd-v2/eval/results/uncert_probe_v1a.json"

def build_inputs(tok, prompt, response):
    msgs = [{"role":"user","content":prompt}]
    p_out = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    p_ids = p_out["input_ids"] if isinstance(p_out, dict) or hasattr(p_out, "input_ids") else p_out
    r_ids = tok(response, return_tensors="pt", add_special_tokens=False).input_ids
    return torch.cat([p_ids, r_ids], dim=1), p_ids.shape[1], r_ids.shape[1]

@torch.no_grad()
def featurize(model, tok, prompt, response, device):
    full, plen, rlen = build_inputs(tok, prompt, response)
    full = full.to(device)
    if rlen < 2: return None
    if rlen > 512:
        full = full[:, :plen+512]; rlen = 512
    out = model(full, output_hidden_states=True)
    hs = out.hidden_states[-1][0]                     # [T, H]
    hs_last = hs[plen+rlen-1].float().cpu().numpy()   # final response token
    hs_mean = hs[plen:plen+rlen].float().mean(0).cpu().numpy()
    return hs_last.astype(np.float32), hs_mean.astype(np.float32)

def main():
    t0 = time.time()
    data = json.load(open(DATA))["items"]
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    print(f"[v1a] model loaded in {time.time()-t0:.1f}s", flush=True)

    X_last, X_mean, y = [], [], []
    for i, it in enumerate(data):
        r = featurize(model, tok, it["prompt"], it["response"], device)
        if r is None: continue
        X_last.append(r[0]); X_mean.append(r[1])
        y.append(1 if it["correct"] else 0)
    X_last = np.stack(X_last); X_mean = np.stack(X_mean); y = np.array(y)
    print(f"[v1a] features extracted: n={len(y)} pos={int(y.sum())} neg={int(len(y)-y.sum())}", flush=True)

    # free model memory
    del model; torch.cuda.empty_cache()

    # Out-of-fold predictions with LogReg and MLP on hs_last
    def oof_predict(clf_factory, X, name):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        pred_correct = np.zeros(len(y), dtype=np.float32)   # P(correct)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = clf_factory()
            clf.fit(sc.transform(X[tr]), y[tr])
            pred_correct[te] = clf.predict_proba(sc.transform(X[te]))[:,1]
        auc = roc_auc_score(y, pred_correct)
        print(f"  {name}: OOF AUC={auc:.3f}", flush=True)
        return pred_correct, auc

    results = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y)-y.sum()),
               "baseline_assertion_asym": -0.009, "bar_assertion_asym": -0.15}

    def evaluate_gate(p_correct, name):
        # Sweep thresholds; the "gate" asserts iff predicted correct prob > thr.
        # Goal: minimize assertion_asym = p_assert(wrong) - p_assert(correct).
        best = None
        for thr in np.linspace(0.05, 0.95, 91):
            assert_flag = (p_correct > thr).astype(int)
            # p_assert on wrong items and on correct items
            n_w = int((y == 0).sum()); n_c = int((y == 1).sum())
            p_w = float(assert_flag[y==0].mean()) if n_w else 0.0
            p_c = float(assert_flag[y==1].mean()) if n_c else 0.0
            asym = p_w - p_c
            # valid if we retain a reasonable fraction of assertions on correct items
            if best is None or asym < best["asym"]:
                best = {"thr": float(thr), "p_assert_wrong": p_w,
                        "p_assert_correct": p_c, "asym": asym}
        # Also report fixed thr=0.5
        at_05 = None
        assert_flag = (p_correct > 0.5).astype(int)
        at_05 = {
            "thr": 0.5,
            "p_assert_wrong": float(assert_flag[y==0].mean()),
            "p_assert_correct": float(assert_flag[y==1].mean()),
            "asym": float(assert_flag[y==0].mean() - assert_flag[y==1].mean()),
        }
        # Balanced: require p_assert_correct >= 0.70 (do not over-hedge correct answers)
        bal = None
        for thr in np.linspace(0.05, 0.95, 91):
            assert_flag = (p_correct > thr).astype(int)
            p_w = float(assert_flag[y==0].mean())
            p_c = float(assert_flag[y==1].mean())
            if p_c >= 0.70:
                asym = p_w - p_c
                if bal is None or asym < bal["asym"]:
                    bal = {"thr": float(thr), "p_assert_wrong": p_w,
                           "p_assert_correct": p_c, "asym": asym}
        bal_str = "none" if bal is None else f"{bal['asym']:.3f}@{bal['thr']:.2f}"
        print(f"  {name}: best_asym={best['asym']:.3f} @thr={best['thr']:.2f}  "
              f"thr=0.5 asym={at_05['asym']:.3f}  balanced(p_c>=0.70)={bal_str}",
              flush=True)
        return {"min": best, "thr_0.5": at_05, "balanced_pc_0.70": bal}

    for name, X in [("hs_last", X_last), ("hs_mean", X_mean)]:
        for clf_name, factory in [
            ("logreg",   lambda: LogisticRegression(max_iter=2000, C=1.0)),
            ("logreg_C0.1", lambda: LogisticRegression(max_iter=2000, C=0.1)),
            ("mlp_32",   lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=0)),
        ]:
            p, auc = oof_predict(factory, X, f"{name}+{clf_name}")
            gate = evaluate_gate(p, f"{name}+{clf_name}")
            results[f"{name}__{clf_name}"] = {"auc": auc, "gate": gate}

    # Verdict: does any configuration clear the bar on a balanced operating point?
    def best_balanced(res):
        b = None
        for k, v in res.items():
            if not isinstance(v, dict) or "gate" not in v: continue
            bal = v["gate"].get("balanced_pc_0.70")
            if bal is None: continue
            if b is None or bal["asym"] < b[1]["asym"]:
                b = (k, bal)
        return b
    bb = best_balanced(results)
    if bb:
        results["best_balanced"] = {"config": bb[0], **bb[1]}
        print(f"[v1a] BEST BALANCED: {bb[0]} asym={bb[1]['asym']:.3f} "
              f"(bar: -0.15, baseline: -0.009)", flush=True)
        results["gate_pass"] = bool(bb[1]["asym"] <= -0.15)
    else:
        results["best_balanced"] = None
        results["gate_pass"] = False
        print("[v1a] no balanced operating point achieved p_assert_correct>=0.70", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT,"w"), indent=2, default=float)
    print(f"[v1a] DONE gate_pass={results['gate_pass']} -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
