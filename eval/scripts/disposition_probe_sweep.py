"""Per-axis hs_last probe sweep across the 4 in-scope DDv2 Chef disposition axes
(plus factual_accuracy as the capability anchor).

Scope (locked with user): Humble (hedging_appropriateness),
Deliberate (pedagogical_framing), Adversarial-self (self_verification),
Persistent (completeness). Eager / Curious / Self-Improving are OUT for Chef.

Method:
  For each axis:
    1. Binarize the v4 rubric score: high = score >= 4, low = score <= 3.
    2. Train LogReg(C=0.1) on hs_last features extracted from step16 baseline (n=100).
    3. 5-fold stratified CV on step16 -> per-axis CV AUC (within-distribution ceiling).
    4. Train on full step16, predict on fresh 193 -> per-axis transfer AUC.
  Cross-axis sanity:
    - Pearson correlation between predicted P(high) for each pair of axes.
    - If pedagogical and factual correlate >0.8 the probe is "good answer" with extra
      steps (CORE_CLAIM separability test).

Inputs (all cached, no GPU work):
  - dd-v2/eval/results/v1_full_step16_features.npz   (hs_last + y_factual labels)
  - dd-v2/eval/results/v1_full_v2_fresh_features.npz (hs_last for 193 fresh items)
  - dd-v2/eval/results/judge_v4_step16.json          (v4-scored 100 step16 items)
  - dd-v2/eval/results/judge_v4_fresh.json           (v4-scored 193 fresh items)

Output:
  - dd-v2/eval/results/disposition_probe_sweep.json
"""
import json, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = '/workspace/persistent/dd-v2/eval/results'
STEP16_FEATS = f'{ROOT}/v1_full_step16_features.npz'
FRESH_FEATS  = f'{ROOT}/v1_full_v2_fresh_features.npz'
STEP16_V4    = f'{ROOT}/judge_v4_step16.json'
FRESH_V4     = f'{ROOT}/judge_v4_fresh.json'
OUT          = f'{ROOT}/disposition_probe_sweep.json'

AXES = ['factual_accuracy', 'hedging_appropriateness', 'pedagogical_framing',
        'self_verification', 'completeness']
DISPOSITION = {
    'factual_accuracy':       'capability_anchor',
    'hedging_appropriateness':'humble',
    'pedagogical_framing':    'deliberate',
    'self_verification':      'adversarial_self',
    'completeness':           'persistent',
}

def load_v4(path):
    d = json.load(open(path))['items']
    out = {a: [] for a in AXES}
    valid_idx = []
    for i, it in enumerate(d):
        s = it['scores']
        if 'error' in s:
            continue
        for a in AXES:
            out[a].append(int(s[a]))
        valid_idx.append(i)
    return {a: np.array(out[a], dtype=int) for a in AXES}, np.array(valid_idx, dtype=int)

def binarize(scores, high_thr=4):
    return (scores >= high_thr).astype(int)

def cv_auc(X, y, n_splits=5, seed=0):
    if y.sum() < 2 or (len(y)-y.sum()) < 2:
        return None, None
    skf = StratifiedKFold(n_splits=min(n_splits, int(min(y.sum(), len(y)-y.sum()))),
                          shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, oof)), oof

def transfer_auc(X_tr, y_tr, X_te, y_te):
    if y_tr.sum() < 2 or (len(y_tr)-y_tr.sum()) < 2:
        return None, None
    if y_te.sum() < 2 or (len(y_te)-y_te.sum()) < 2:
        return None, None
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X_tr), y_tr)
    p = clf.predict_proba(sc.transform(X_te))[:, 1]
    return float(roc_auc_score(y_te, p)), p

def main():
    s16 = np.load(STEP16_FEATS)
    fr  = np.load(FRESH_FEATS)
    X_tr = s16['hs_last'].astype(np.float32)
    X_te = fr['hs_last'].astype(np.float32)
    print(f'[sweep] step16 features: {X_tr.shape}  fresh features: {X_te.shape}')

    s16_scores, s16_valid = load_v4(STEP16_V4)
    fr_scores, fr_valid   = load_v4(FRESH_V4)
    print(f'[sweep] step16 valid v4 items: {len(s16_valid)}/{X_tr.shape[0]}')
    print(f'[sweep] fresh   valid v4 items: {len(fr_valid)}/{X_te.shape[0]}')

    # restrict feature matrices to v4-valid rows
    X_tr_v = X_tr[s16_valid]
    X_te_v = X_te[fr_valid]

    results = {
        'n_step16': int(len(s16_valid)),
        'n_fresh':  int(len(fr_valid)),
        'high_thr': 4,
        'classifier': 'LogReg(C=0.1) on standardized hs_last',
        'note_bias': ('v4 = v2 response-blind 2-pass + v1 5-axis. Within-condition '
                      'ranking (used here) is much less affected by length-halo than '
                      'cross-condition comparison. Caveat documented.'),
        'axes': {},
    }

    oof_preds = {}
    transfer_preds = {}
    for axis in AXES:
        y_tr_full = binarize(s16_scores[axis])
        y_te_full = binarize(fr_scores[axis])
        n_pos_tr = int(y_tr_full.sum()); n_neg_tr = int(len(y_tr_full)-n_pos_tr)
        n_pos_te = int(y_te_full.sum()); n_neg_te = int(len(y_te_full)-n_pos_te)

        cv, oof = cv_auc(X_tr_v, y_tr_full)
        tr, ptr = transfer_auc(X_tr_v, y_tr_full, X_te_v, y_te_full)

        if oof is not None: oof_preds[axis] = oof
        if ptr is not None: transfer_preds[axis] = ptr

        ax_block = {
            'disposition': DISPOSITION[axis],
            'n_pos_step16': n_pos_tr, 'n_neg_step16': n_neg_tr,
            'n_pos_fresh':  n_pos_te, 'n_neg_fresh':  n_neg_te,
            'cv_auc_step16': cv,
            'transfer_auc_fresh': tr,
            'pass_cv':       bool(cv is not None and cv >= 0.65),
            'pass_transfer': bool(tr is not None and tr >= 0.60),
        }
        results['axes'][axis] = ax_block
        cv_s = f'{cv:.3f}' if cv is not None else 'N/A'
        tr_s = f'{tr:.3f}' if tr is not None else 'N/A'
        print(f'  {axis:25s} ({DISPOSITION[axis]:20s})  '
              f'pos={n_pos_tr:3d}/{n_pos_te:3d}  '
              f'cv_auc={cv_s}  transfer={tr_s}')

    # cross-axis correlation on OOF predictions (separability check)
    cross = {}
    keys = list(oof_preds.keys())
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            r = float(np.corrcoef(oof_preds[a], oof_preds[b])[0, 1])
            cross[f'{a}__{b}'] = r
    results['cross_axis_oof_correlation'] = cross
    flagged = {k: v for k, v in cross.items() if abs(v) >= 0.8}
    results['cross_axis_flagged'] = flagged
    if flagged:
        print('[sweep] WARN: high cross-axis correlation (probe may collapse to "good answer"):')
        for k, v in flagged.items():
            print(f'  {k}: r={v:.3f}')

    # gate
    n_axes_total = len(AXES) - 1  # excluding factual anchor
    n_pass = sum(1 for a in AXES if a != 'factual_accuracy'
                 and results['axes'][a]['pass_cv']
                 and results['axes'][a]['pass_transfer'])
    results['gate'] = {
        'criterion': 'each disposition axis: cv_auc>=0.65 AND transfer_auc>=0.60; cross-axis corr<0.80',
        'n_disposition_axes_passing': int(n_pass),
        'n_disposition_axes_total':   int(n_axes_total),
        'pass_all':  bool(n_pass == n_axes_total and not flagged),
        'pass_some': bool(n_pass >= 1),
    }
    print(f'[sweep] {n_pass}/{n_axes_total} disposition axes pass; '
          f'cross-axis flagged={len(flagged)}; '
          f'pass_all={results["gate"]["pass_all"]}')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, 'w'), indent=2, default=float)
    print(f'[sweep] DONE -> {OUT}')

if __name__ == '__main__':
    main()
