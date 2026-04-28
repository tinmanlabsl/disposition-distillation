"""T1-lite: hard-gated hedge-prefix sidecar evaluation.

Reuses everything v1_full_v2 cached:
  - fresh prompts          (v1_full_fresh_prompts.json)
  - fresh hs_last features (v1_full_v2_fresh_features.npz)
  - fresh Gemma responses  (v1_full_fresh_responses.json)
  - fresh gold checklists  (v1_full_v2_gold.json)
  - per-claim coverage labels (v1_full_v2_results.json -> 'fresh_labels' / 'fresh_p')
  - trained gate (refit here from step16 features for clarity)

Behavior:
  1. Refit LogReg(C=0.1) gate on step16 hs_last features (the v1a champion).
  2. Predict p_correct on the 193 fresh items.
  3. Choose threshold = balanced operating point on STEP16 (NOT cheating: pick
     threshold from training data, not from fresh labels).
  4. assert_flag = (p_correct > thr); hedge otherwise. Content unchanged.
  5. Compute on FRESH labels:
       - pre_gate_asym  = baseline assertion_asym ≈ 0
         (everything asserted by default; bar = -0.009 from v1a)
       - post_gate_asym = p_assert(wrong) - p_assert(correct)
       - coverage       = p_assert(correct)   (must stay >= 0.70)
       - accuracy       = unchanged (sanity)
  6. Also report numbers if we *cheat* and pick the threshold on fresh —
     gives an upper bound on what a perfectly calibrated threshold could yield.

Pass: post_gate_asym <= -0.15 AND coverage >= 0.70 with the
TRAIN-CHOSEN threshold (no cheating).
"""
import json, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT     = "/workspace/persistent/dd-v2/eval/results"
STEP16   = f"{ROOT}/baseline_gemma4_v2_from_step16.json"
FRESH_F  = f"{ROOT}/v1_full_v2_fresh_features.npz"
RESULTS  = f"{ROOT}/v1_full_v2_results.json"
STEP16_F = f"{ROOT}/v1_full_step16_features.npz"
OUT      = f"{ROOT}/t1_lite.json"

def main():
    # ---- step16 features for training gate
    s16 = np.load(STEP16_F)
    X_tr = s16["hs_last"].astype(np.float32)
    y_tr = s16["y"].astype(int)
    print(f"[t1] step16 train: n={len(y_tr)} pos={int(y_tr.sum())}")

    # ---- fresh features and labels
    fr = np.load(FRESH_F)
    X_te = fr["hs_last"].astype(np.float32)
    res = json.load(open(RESULTS))
    y_te = np.array(res["fresh_labels"], dtype=int)
    p_te_v1full = np.array(res.get("fresh_p_logreg_C0.1", res.get("fresh_p", [])), dtype=float)
    print(f"[t1] fresh test: n={len(y_te)} pos={int(y_te.sum())}")

    # ---- refit gate (same recipe as v1a champion)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X_tr), y_tr)
    p_tr = clf.predict_proba(sc.transform(X_tr))[:,1]
    p_te = clf.predict_proba(sc.transform(X_te))[:,1]

    def asym(p, y, thr):
        f = (p > thr).astype(int)
        nw, nc = int((y==0).sum()), int((y==1).sum())
        pw = float(f[y==0].mean()) if nw else 0.0
        pc = float(f[y==1].mean()) if nc else 0.0
        return {"thr":float(thr),"p_assert_wrong":pw,"p_assert_correct":pc,"asym":pw-pc}

    # threshold chosen on TRAIN (no peek) at balanced p_c >= 0.70
    train_pick = None
    for thr in np.linspace(0.05, 0.95, 91):
        m = asym(p_tr, y_tr, thr)
        if m["p_assert_correct"] >= 0.70:
            if train_pick is None or m["asym"] < train_pick["asym"]:
                train_pick = m
    if train_pick is None:
        train_pick = asym(p_tr, y_tr, 0.5)
        print("[t1] WARN: no train threshold met p_c>=0.70, fell back to 0.5")
    thr = train_pick["thr"]
    print(f"[t1] train-chosen thr={thr:.3f}  train_asym={train_pick['asym']:.3f}")

    test_at_train_thr = asym(p_te, y_te, thr)

    # cheating upper bound: best balanced thr on fresh
    cheat = None
    for t in np.linspace(0.05, 0.95, 91):
        m = asym(p_te, y_te, t)
        if m["p_assert_correct"] >= 0.70:
            if cheat is None or m["asym"] < cheat["asym"]:
                cheat = m

    pre_gate = {"thr":0.0,"p_assert_wrong":1.0,"p_assert_correct":1.0,"asym":0.0}

    out = {
        "n_fresh": int(len(y_te)),
        "n_fresh_pos": int(y_te.sum()),
        "train_threshold": train_pick,
        "fresh_at_train_threshold": test_at_train_thr,
        "fresh_oracle_threshold": cheat,
        "pre_gate_baseline": pre_gate,
        "bar": {"asym": -0.15, "coverage_min": 0.70},
        "pass_honest": bool(test_at_train_thr["asym"] <= -0.15
                            and test_at_train_thr["p_assert_correct"] >= 0.70),
        "pass_oracle": bool(cheat is not None
                            and cheat["asym"] <= -0.15
                            and cheat["p_assert_correct"] >= 0.70),
    }
    print(f"[t1] HONEST: asym={test_at_train_thr['asym']:.3f}  "
          f"cov={test_at_train_thr['p_assert_correct']:.3f}  pass={out['pass_honest']}")
    if cheat:
        print(f"[t1] ORACLE: asym={cheat['asym']:.3f}  cov={cheat['p_assert_correct']:.3f}  "
              f"thr={cheat['thr']:.3f}  pass={out['pass_oracle']}")
    json.dump(out, open(OUT,"w"), indent=2, default=float)
    print(f"[t1] DONE -> {OUT}")

if __name__ == "__main__":
    main()
