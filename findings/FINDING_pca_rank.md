# Finding: DD style shift is distributed, not a sharp low-rank attractor

**Date:** 2026-04-08
**Experiment:** `dd-v2/eval/pca_rank.py`
**Raw data:** `dd-v2/eval/results/pca_rank.json`
**Log:** `dd-v2/eval/logs/pca_rank.log`

## Question

Open question: is the DD→base deflection a coherent, low-rank direction that can be isolated (Gemma's GR-DPO) or a distributed multi-subspace shift (GLM's CCA-steering)? This experiment measures the rank of the deflection directly, independent of any gold-checklist scoring.

## Method

- Base: Gemma 4 E2B. DD adapter: `chef_dpo/final`.
- Same prompt set used in step16 held-out region (80 prompts).
- For each prompt, generate 200 tokens with base (via `disable_adapter`) and DD. Extract residual-stream activations at layer −8. Mean-pool over generated tokens → one 1536-d vector per prompt per model.
- Compute SVD-based PCA on three matrices:
  - `base` (80 × 1536): pure base content representation
  - `dd` (80 × 1536): DD representation
  - `diff = dd − base` (80 × 1536): the style-shift direction
- Report number of principal components needed for 50/80/90/95/99% variance explained.

## Results

| Matrix | 50% | 80% | 90% | 95% | 99% | Top-1 EVR |
|---|---|---|---|---|---|---|
| base | 6 | 19 | 32 | 45 | 68 | 0.168 |
| dd | 6 | 19 | 32 | 46 | 67 | 0.167 |
| **diff (dd − base)** | **10** | **32** | **47** | **59** | **74** | **0.110** |

### Interpretation of the numbers

1. **Base and DD geometries are nearly identical.** Same rank at every threshold; top-1 EVR within 0.001. The adapter does not fundamentally warp the representational manifold — it *shifts* it.

2. **The shift (diff) is mid-rank, not a sharp attractor.**
   - Top PC of diff captures only **11%** of variance — there is no single "disposition direction" to extract (Zou 2023–style Representation Engineering single-vector steering is ruled out here).
   - 90% of the shift lives in 47 dimensions — low-rank in absolute terms (3% of 1536), but ~50% more spread than the base content representation (47 vs 32).
   - The 1-PC/2-PC/3-PC energy tail (0.110 → 0.089 → 0.060) is gentle, not dominated. A clean null-space projection would need to remove ≥30 dimensions to capture most of the style, and those same 30 dimensions overlap heavily with dimensions base uses for content (since diff's rank profile tracks base's).

## What this tells us about each candidate path

### 1. Gemma's Gradient-Reversal DPO — BORDERLINE / HIGH RISK
GR-DPO as specified assumes the style lives in a projectable null-space. With diff top-1 at only 11% and 90% needing 47 dimensions, null-space projection would have to either (a) leave most of the style untouched, or (b) damage the content-bearing subspace. **Not viable in its pure form.** Survivable only with a learned multi-subspace projector, which is no longer what "gradient reversal" cheaply buys you.

### 2. GLM's CCA-aligned attention-head steering — FAVORED
CCA is built precisely for the regime where two matrices share a multi-dimensional affine relationship with no dominant direction. The diff rank profile is consistent with what CCA is designed to find: a set of ~30-50 canonical correlation pairs between base and DD. This is the geometric path with the cleanest match to the data.

### 3. VGSD (Verifier-Gated Self-Distillation) — UNBLOCKED, GEOMETRY-INDEPENDENT
VGSD does not depend on PCA rank at all — it samples from base best-of-N, scores against gold, and DPOs base against its own worst samples. The PCA result neither helps nor hurts it. It remains the fallback path and is being staged for overnight feasibility sampling.

## Decision (tentative, pending VGSD feasibility data)

- **Kill:** Pure GR-DPO null-space projection. Single-vector representation-engineering steering.
- **Prefer:** CCA-aligned steering as the leading geometric path, if the next experiment commits to a geometric approach.
- **Parallel hedge:** VGSD feasibility sampling tonight. If base has ≥1 good sample in 8 on ≥50% of prompts, VGSD becomes the lowest-risk path regardless of geometry.

## What this adds to the paper

This is the second independent falsification in the DD-v2 method-validation thread. Combined with the α-sweep finding, we now have mechanistic evidence that:

1. DD fine-tuning does not produce a single coherent "disposition vector" in residual-stream geometry (contra Representation Engineering optimism).
2. DD fine-tuning does not alter the representational manifold's intrinsic dimensionality — it translates it along a distributed mid-rank shift.
3. Per-token logit mixing fails because the shift is distributed (α-sweep U-curve).
4. Per-vector steering would fail for the same reason — there is no dominant direction to steer along.

The positive contribution is: **distributed low-rank shift with no sharp attractor** is a characterization worth reporting, and it predicts which intervention families can and cannot work on this class of SFT/DPO-trained disposition adapters.

## Files
- `dd-v2/eval/pca_rank.py` — experiment
- `dd-v2/eval/results/pca_rank.json` — full EVR + rank table
- `dd-v2/eval/logs/pca_rank.log` — run log
- `dd-v2/findings/FINDING_pca_rank.md` — this doc

## Next step
VGSD base-sampling feasibility probe (launching now): 8 samples × 100 gold prompts on base Gemma 4 E2B, scored via v3 judge, to measure base's "sometimes knows it" ceiling. Result → `FINDING_vgsd_feasibility.md`.
