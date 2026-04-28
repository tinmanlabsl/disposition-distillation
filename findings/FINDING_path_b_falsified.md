# Finding: Path B (Inference-Time Head Tempering) FALSIFIED Across Models and Attribution Methods

**Date:** 2026-04-08
**Status:** CLOSED — path B falsified on Humble across Gemma 4 E2B and Qwen3.5-0.8B, under both magnitude and directional attribution.
**Consolidates:**
- `FINDING_path_b1_attribution.md` — B.1 attribution results (both models)
- `FINDING_path_b2_gemma4_negative.md` — Gemma Humble + Adv-self B.2 sweeps
- `FINDING_path_b2_qwen35_humble.md` — Qwen3.5 Humble B.2 sweep + replication
**Raw data:**
- `dd-v2/eval/results/path_b2_tempering_gemma4_summary.json`
- `dd-v2/eval/results/path_b2_gemma4_adversarial_samples.json`
- `dd-v2/eval/results/path_b2_qwen35_0.8b_humble_summary.json`
- `dd-v2/eval/results/path_b2_qwen35_0.8b_humble_rep2_samples.json`
- `dd-v2/eval/results/path_b_attribution_qwen35_0.8b_humble_directional.json`

## The claim that was tested

From `CORE_CLAIM_ddv2.md`:
> Disposition is a low-rank, inference-time-steerable axis separable from content. Each of the 7 dispositions can be addressed by damping a small set of attention heads (~5) identified via activation-based attribution on a confident-wrong vs confident-right cohort split. The artifact is `base_model → {disposition: [(layer, head, λ), ...]}` composed at inference with no per-specialist retraining.

The falsifiable dual criterion: damping top-K heads must (a) shift the disposition metric in the target direction, (b) preserve gold-checklist coverage.

## What we ran

Four experiments over two base models and three method variants on the chef 100-prompt harness:

### 1. Gemma 4 E2B + Humble + magnitude attribution
- **B.1:** top-5 heads by Cohen's d on per-head L2 activation norms, cr vs cw cohort. Max d = 0.60. Distributed across L3, L16, L17, L18, L20.
- **B.2:** 4-λ sweep ∈ {1.0, 0.9, 0.8, 0.5}. Greedy generation (bug — original baseline was also greedy for Gemma, so internal comparison held).
- **Result:** assertion rate frozen 0.89→0.87 across all λ. Hallucination rate noisy flat (0.85→0.80). cw cell 29→33 (moved *wrong direction*). Accuracy dropped 7pts (0.66→0.58→0.59). Content moves with intervention, disposition does not.
- **Verdict:** null on disposition, modest content damage.

### 2. Gemma 4 E2B + Adversarial-self + magnitude attribution
- **Cohort split:** reused `judge_rubric.json` self_verification axis from Step 16 baseline (advH ≥4: n=11, advL ≤2: n=69). Deliberate, Curious were not attributable on this harness — pedagogical_framing 98/100 saturated high, certainty 0/100 ambiguous.
- **B.1:** stronger signal than Humble. Max |d| = 0.875 at L23-H0. **L11 cluster of 4 heads in top-10.** This was the best localized signal we found on Gemma.
- **B.2:** damped top-5 (L23-H0, L11-H5, L11-H0, L11-H6, L15-H4, all negative d → damping should *increase* self-verification). Partial sweep: λ=1.0 and λ=0.9 before kill.
- **Result:** self_verif_mean moved **wrong direction** (3.69 → 3.58). High-verif cohort shrank 82→75. Coverage flat (0.474 → 0.479).
- **Killed after λ=0.9** (pattern confirmed wrong-direction, saved ~$5 + 1h).
- **Verdict:** null/wrong-direction disposition, flat content. The strongest localized Gemma signal still doesn't survive causal intervention.

### 3. Qwen3.5-0.8B + Humble + magnitude attribution
- **B.1 (prior work):** top-5 heads with **L23 cluster** (3 of 5 in final attention layer). Max d = 0.557. Qualitatively different signal shape from Gemma — late-layer concentration.
- **B.2 run 1 (bug):** greedy decoding, `enable_thinking` not set. Caused distribution drift vs sampled baseline (cr 64→17, cw 18→78). **Killed and corrected.**
- **B.2 run 2 (corrected gen params):** Qwen3.5 official recommended sampling `temp=1.0, top_p=1.0, top_k=20`, `enable_thinking=False`, seeded `torch.manual_seed(1000+i)` per prompt for cross-λ comparability.
- **Result:**

| λ | cr | hr | cw | hw | acc | assert | hall |
|---|---|---|---|---|---|---|---|
| 1.0 | 19 | 2 | 72 | 7  | 0.210 | 0.910 | 0.911 |
| 0.9 | 14 | 3 | 75 | 8  | 0.170 | 0.890 | 0.904 |
| 0.8 | 12 | 3 | 75 | 10 | 0.150 | 0.870 | 0.882 |
| 0.5 | **22** | 1 | **63** | 14 | **0.230** | 0.850 | **0.818** |

λ=0.9 and 0.8 showed the same wrong-direction content-damage pattern as Gemma (cr shrinks, cw flat). **λ=0.5 broke the pattern** — cr recovered above baseline, cw dropped 9, hall dropped 9.3pts, cw→hw migration visible (confident-wrong became hedged-wrong). First apparent positive signal in the project.

### 4. Qwen3.5-0.8B + Humble + magnitude — REPLICATION
- Same script, `torch.manual_seed(2000+i)`. Single λ=0.5 cell.
- **Result:** cr=12, cw=77, acc=0.130, hall=0.885. Every pre-declared "sampling artifact" criterion hit.
- **Verdict:** the main run's λ=0.5 was an RNG draw, not a real effect. Replication reverted to the content-damage pattern of λ=0.8.

### 5. Qwen3.5-0.8B + Humble + DIRECTIONAL attribution
- Method: capture full per-head activation vectors (not just L2 norms). Compute cohort mean vectors µ_cr, µ_cw. Direction `v = µ_cw − µ_cr`. Project each sample onto `v`, compute Cohen's d over projections.
- **Result:** top-10 d_directional values 1.73, 1.58, 1.49, 1.47, 1.43, 1.41, 1.40, 1.31, 1.30, 1.29.
- **Caveat — statistical artifact.** `cos(µ_cr, µ_cw)` on every top head ≈ **1.0** (0.998, 0.987, 0.994, 0.967, 0.996, 0.966, 0.958, 0.998, 0.991, 0.998). The cohort means are nearly collinear. The large d_directional is a variance artifact from projecting onto a tiny orthogonal direction — individual sample variance along that direction is near zero, so pooled std → 0, inflating Cohen's d.
- The heads directional attribution ranks to the top also cluster in L11/L19/L23 — same layers as magnitude attribution. It finds the same circuit, ranked differently, not a hidden axis.
- **Verdict:** directional attribution does not reveal a separable axis that magnitude missed. The cr and cw cohorts are not linearly separable in per-head activation space at all — neither in magnitude nor in direction.

## Consolidated pattern

Across 5 experimental variants:

| # | Model | Disposition | Method | Disposition metric moves? | Content preserved? |
|---|---|---|---|---|---|
| 1 | Gemma 4 E2B | Humble | magnitude | ❌ frozen | ❌ 7pt drop |
| 2 | Gemma 4 E2B | Adversarial-self | magnitude | ❌ wrong direction | ✅ flat |
| 3 | Qwen3.5-0.8B | Humble | magnitude | ⚠️ non-monotonic | ⚠️ drops then recovers |
| 4 | Qwen3.5-0.8B | Humble (replication) | magnitude | ❌ wrong direction | ❌ worse |
| 5 | Qwen3.5-0.8B | Humble | directional | N/A — no separable axis found | N/A |

**Zero clean positives. The apparent #3 positive was single-seed RNG and did not replicate under #4.**

## What this falsifies

The narrow, falsifiable version of the DD v2 Path B claim as operationalized:

> For a given base model and disposition, one can (a) identify the top-K attention heads associated with the disposition via a forward-pass cohort attribution (magnitude or directional), and (b) damp those heads by a scalar λ<1 at inference time to cause a causal shift in the disposition metric while preserving content.

This is falsified for the Humble disposition on Gemma 4 E2B and Qwen3.5-0.8B, and for the Adversarial-self disposition on Gemma 4 E2B. Both magnitude and directional attribution were tried on Qwen3.5. Two different architectures (dense Gemma, hybrid DeltaNet Qwen3.5) and two different signal shapes (distributed Gemma, L23-clustered Qwen3.5) produce the same null under causal intervention.

## What this does NOT falsify

1. **Inference-time disposition shaping in general.** This test used a specific intervention (o_proj input damping with scalar λ) at a specific location (attention output) using a specific attribution (cohort-mean cr vs cw). Alternative interventions at other locations (v_proj, MLP down_proj, residual stream), other schedules (LoRA-style low-rank adapter at inference), or other attribution methods (gradient-based, activation patching) are untested.
2. **Disposition separability at larger scales.** Both tested models are sub-2.4B. The claim that circuits become more localized at ~7B+ has not been evaluated here — compute and budget prevented this.
3. **Other dispositions.** Five of the seven dispositions in the taxonomy were not fully tested. Adversarial-self was partially tested on Gemma only. Deliberate was not attributable on chef (rubric saturation). Curious, Eager, Persistent, Self-Improving were not tested at all.
4. **DD v2 training-side interventions.** The falsification applies to Path B (inference-time). DD v2 Path A (training-side) is tested in separate runs and has its own finding docs (DD v2 Chef imitation pipeline failed on a different construction — not the same falsification).

## What this implies

The narrow version of DD v2 Path B — "simple recipe, same method everywhere" — is not viable as the research program's center of gravity. Several honest reframings are possible:

**Reframe 1: model-scale hypothesis.** Disposition circuits may be too diffuse/entangled at 0.6–2.4B scale to isolate with forward-pass attribution + scalar damping. Would require testing at 4–7B+ to falsify or confirm. Cost: ~$20–40 per base model for B.1+B.2 at larger scale.

**Reframe 2: intervention class hypothesis.** Scalar head damping may be the wrong primitive at this scale. Low-rank LoRA adapter at inference (rank-1 or rank-2, per disposition) is a more expressive intervention that could capture distributed signal. Significantly more work than Path B but plausibly effective.

**Reframe 3: disposition-is-not-mechanistic hypothesis.** Disposition may not correspond to separable mechanistic circuits at all — it may be an emergent property of the full forward pass, shaped by training data distribution rather than identifiable feature directions. If true, the only interventions that can reliably shift disposition are training-level (which DD v1 SFT + DPO have already been shown to make performative-only on sub-2B per the comprehensive DD findings).

**Reframe 4: accept falsification at this scale, pivot research direction.** Document the cross-model negative as the main DD v2 result. Reframe the program around: "disposition distillation at sub-2B is not a separable low-rank axis under the interventions we tested; the shaping tools that work are full training or prompt engineering, both of which have their own limits at this scale." This is a real, publishable negative that contributes to the interpretability literature by narrowing the space of viable claims.

## Decision recorded

Path B is closed as the primary DD v2 method as of 2026-04-08. No further B.1/B.2 sweeps on Humble or Adversarial-self on currently-tested models. Next-step options under consideration (not yet committed):
- Directional attribution retry on Gemma (for symmetry, ~10 min, $0) — optional
- Run B.1+B.2 on Qwen3-4B or similar larger model — decision deferred
- Pivot to writing consolidated DD pivot summary (DD_Pivot_reason.md)

## Files

- `dd-v2/eval/path_b_attribution.py` — shared magnitude attribution (Gemma Humble + Qwen3.5 Humble)
- `dd-v2/eval/path_b_attr_gemma4_adv.py` — Gemma Adv-self attribution
- `dd-v2/eval/path_b2_tempering.py` — Gemma Humble B.2
- `dd-v2/eval/path_b2_gemma4_adv.py` — Gemma Adv-self B.2 (killed after λ=0.9)
- `dd-v2/eval/path_b2_qwen35_humble.py` — Qwen3.5 Humble B.2
- `dd-v2/eval/path_b2_qwen35_humble_rep.py` — Qwen3.5 Humble λ=0.5 replication
- `dd-v2/eval/path_b1_directional_qwen35.py` — Qwen3.5 directional attribution
- `dd-v2/eval/logs/path_b2_gemma4.log`
- `dd-v2/eval/logs/path_b2_adv.log`
- `dd-v2/eval/logs/path_b2_qwen35.log`
- `dd-v2/eval/logs/path_b2_qwen35_rep.log`
- `dd-v2/eval/logs/path_b1_directional_qwen35.log`
