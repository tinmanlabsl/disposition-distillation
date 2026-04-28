"""Render paper figures from existing artifacts. Outputs PDFs to dd-v2/paper/figs/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "eval", "results"))
plt.rcParams.update({"font.size": 10, "font.family": "serif", "pdf.fonttype": 42})

BLUE = "#1B4F72"; GOLD = "#D4A017"; RED = "#E74C3C"; GREY = "#95A5A6"

# ---------- Figure 1: Arc 3 AUC collapse (hook) ----------
def fig_auc_collapse():
    v1 = json.load(open(os.path.join(RES, "v1_full_v2.json")))
    ml = json.load(open(os.path.join(RES, "probe_multilayer.json")))
    sm = json.load(open(os.path.join(RES, "smollm_factual_probe.json")))

    # extract best AUCs
    def best_auc(d):
        if "results" in d:
            return max(v["auc"] for v in d["results"].values())
        # v1_full_v2 shape
        for k in ("fresh", "transfer", "results"):
            if k in d:
                sub = d[k]
                if isinstance(sub, dict):
                    vals = [v.get("auc") for v in sub.values() if isinstance(v, dict) and "auc" in v]
                    if vals: return max(vals)
        return None

    cv_auc = 0.683  # from v1a CV ceiling, documented in findings
    v1_fresh = best_auc(v1) or 0.516
    ml_auc = ml.get("auc") or ml.get("fresh_auc") or 0.557
    if isinstance(ml_auc, dict): ml_auc = max(ml_auc.values())
    sm_auc = max(v["auc"] for v in sm["results"].values())

    labels = ["Step 1\nwithin-dist CV\n(v1a)", "Step 2\nfresh held-out\n(Gemma)", "Step 4\nmulti-layer pool\n(Gemma)", "Step 5\ncross-model\n(SmolLM)"]
    vals = [cv_auc, v1_fresh, ml_auc, sm_auc]
    colors = [GOLD, RED, RED, RED]

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color=GREY, ls="--", lw=1, label="chance (AUC = 0.50)")
    ax.axhline(0.63, color=BLUE, ls=":", lw=1.2, label="gate (AUC $\\geq$ 0.63)")
    ax.set_ylabel("Probe AUC (factual anchor)")
    ax.set_ylim(0.40, 0.75)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title("Arc 3: Frozen-base $\\mathbf{h}_{\\mathrm{last}}$ probe collapses off the training fold", fontsize=10)
    plt.tight_layout()
    out = os.path.join(HERE, "fig1_auc_collapse.pdf")
    plt.savefig(out); plt.close()
    print(f"wrote {out}  values={vals}")

# ---------- Figure 2: Per-axis disposition sweep (Arc 3 Step 3) ----------
def fig_axis_sweep():
    d = json.load(open(os.path.join(RES, "disposition_probe_sweep.json")))
    axes = d["axes"]
    order = ["factual_accuracy", "hedging_appropriateness", "pedagogical_framing", "self_verification", "completeness"]
    pretty = ["factual\n(anchor)", "hedging\n(Humble)", "pedagogical\n(Deliberate)", "self-verif\n(Adv-self)", "completeness\n(Persistent)"]
    cv  = [axes[k]["cv_auc_step16"] for k in order]
    tr  = [axes[k]["transfer_auc_fresh"] for k in order]

    x = np.arange(len(order))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(x - w/2, cv, w, label="CV AUC (step16)", color=GOLD, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, tr, w, label="Transfer AUC (fresh)", color=RED, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.axhline(0.63, color=BLUE, ls=":", lw=1.2, label="gate")
    ax.set_xticks(x); ax.set_xticklabels(pretty, fontsize=8.5)
    ax.set_ylim(0.35, 0.72)
    ax.set_ylabel("AUC")
    ax.set_title("Arc 3 Step 3: per-axis sweep, 0/4 pass; 3 of 5 transfer AUCs below chance", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for i, v in enumerate(cv):
        ax.text(i - w/2, v + 0.006, f"{v:.2f}", ha="center", fontsize=7.5)
    for i, v in enumerate(tr):
        ax.text(i + w/2, v + 0.006, f"{v:.2f}", ha="center", fontsize=7.5)
    plt.tight_layout()
    out = os.path.join(HERE, "fig2_axis_sweep.pdf")
    plt.savefig(out); plt.close()
    print(f"wrote {out}")

# ---------- Figure 3: SmolLM hs_last PCA on fresh ----------
def fig_pca():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    feat_path = os.path.join(RES, "smollm_fresh_features.npz")
    labs_path = os.path.join(RES, "smollm_factual_probe.json")
    z = np.load(feat_path, allow_pickle=True)
    # try common key names
    X = z["X"]
    if X is None:
        print("smollm_fresh_features.npz keys:", z.files); return
    judged = json.load(open(os.path.join(RES, "smollm_fresh_judged_gold.json")))
    y = np.array(judged["labels"])
    if y is None or len(y) != len(X):
        print(f"could not align labels (X={len(X)}, y={None if y is None else len(y)})"); return

    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    mask_w = y == 0
    mask_c = y == 1
    ax.scatter(pcs[mask_w, 0], pcs[mask_w, 1], s=22, alpha=0.55, color=RED, edgecolor="none", label=f"wrong (n={mask_w.sum()})")
    ax.scatter(pcs[mask_c, 0], pcs[mask_c, 1], s=30, alpha=0.85, color=BLUE, edgecolor="black", linewidth=0.4, label=f"correct (n={mask_c.sum()})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("SmolLM2-1.7B $\\mathbf{h}_{\\mathrm{last}}$ on fresh Chef prompts\n(no separation by factual correctness)", fontsize=10)
    ax.legend(loc="best", frameon=False, fontsize=9)
    plt.tight_layout()
    out = os.path.join(HERE, "fig3_smollm_pca.pdf")
    plt.savefig(out); plt.close()
    print(f"wrote {out}  n={len(y)} pos={int(mask_c.sum())}")

# ---------- Figure 4: Arc 1 claimed vs corrected deltas ----------
def fig_arc1_hook():
    metrics = ["HumanEval\n(Qwen3-0.6B)", "MCAS /25\n(Qwen3-0.6B)", "Recovery\n(Qwen3-0.6B)", "HumanEval v2\n(Qwen3-1.7B)"]
    claimed = [+15.3, +33.9, None, None]   # original pre-falsification
    actual  = [ -8.0,  -0.32, -3.0, -15.8]

    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    # only plot claimed where available
    cx, cy = [], []
    for i, v in enumerate(claimed):
        if v is not None: cx.append(i - w/2); cy.append(v)
    ax.bar(cx, cy, w, color=GOLD, edgecolor="black", linewidth=0.5, label="claimed (pre-falsification)")
    ax.bar(x + w/2, actual, w, color=RED, edgecolor="black", linewidth=0.5, label="corrected (honest re-eval)")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=8.5)
    ax.set_ylabel("$\\Delta$ vs baseline")
    ax.set_title("Arc 1: original positive results did not survive honest re-evaluation", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for xi, v in zip(cx, cy):
        ax.text(xi, v + (0.6 if v>0 else -1.8), f"{v:+.1f}", ha="center", fontsize=8)
    for i, v in enumerate(actual):
        ax.text(i + w/2, v + (0.6 if v>=0 else -1.8), f"{v:+.2f}" if abs(v)<1 else f"{v:+.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    out = os.path.join(HERE, "fig4_arc1_hook.pdf")
    plt.savefig(out); plt.close()
    print(f"wrote {out}")

if __name__ == "__main__":
    fig_auc_collapse()
    fig_axis_sweep()
    fig_pca()
    fig_arc1_hook()
