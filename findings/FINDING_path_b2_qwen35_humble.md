# Finding: Path B.2 Tempering — Qwen3.5-0.8B Humble (Non-Monotonic, Replication Pending)

**Date:** 2026-04-08
**Status:** NEGATIVE — λ=0.5 positive signal did not replicate. With seed offset shifted from 1000→2000, λ=0.5 reverted to the content-damage pattern (cr 12, cw 77, acc 0.130, hall 0.885). The main run's λ=0.5 result was a sampling artifact. Qwen3.5 Humble joins Gemma Humble in the negative column. See replication section below.
**Scripts:**
- `dd-v2/eval/path_b2_qwen35_humble.py` — main 4-λ sweep
- `dd-v2/eval/path_b2_qwen35_humble_rep.py` — λ=0.5 replication with different seed offset
**Raw data:**
- `dd-v2/eval/results/path_b2_qwen35_0.8b_humble_summary.json`
- `dd-v2/eval/results/path_b2_qwen35_0.8b_humble_samples.json`
**Depends on:** `FINDING_path_b2_gemma4_negative.md`, `CORE_CLAIM_ddv2.md`

## Question

Cross-model replication of the Gemma Humble B.2 negative. Does damping the top-5 heads identified by Qwen3.5 B.1 attribution causally shift Humble disposition with content preserved? Qwen3.5's B.1 top-5 concentrates in L23 (3 of 5 heads), structurally different from Gemma's distributed signal — a genuine test of whether localized-signal models behave differently.

## Method

- Base model: `Qwen/Qwen3.5-0.8B` (24 decoder layers, 6 attention + 18 DeltaNet, 8 heads × head_dim 256)
- Harness: identical to Gemma run — 100 chef prompts, gold-checklist coverage judge, ASSERT/HEDGE verbal classifier, DeepSeek V3.2-exp via siliconflow/fp8
- Heads damped (top-5 by Cohen d from `path_b_attribution_qwen35_0.8b.json`):
  - L7-H6 (d=+0.557)
  - L23-H1 (d=+0.544)
  - L19-H2 (d=+0.500)
  - L23-H0 (d=+0.469)
  - L23-H7 (d=+0.442)
- λ sweep: {1.0, 0.9, 0.8, 0.5} via `forward_pre_hook` on `self_attn.o_proj`
- **Gen params (critical — initial run had bug, relaunched):** Qwen3.5 official recommended sampling: `temp=1.0, top_p=1.0, top_k=20, enable_thinking=False`. `torch.manual_seed(1000 + prompt_i)` before each generation — same seed across λ values, different seed per prompt. This makes each (prompt, λ) pair deterministic-but-sampled so λ<1 deltas are the tempering effect, not RNG noise.

### Bug fixed before final run

First attempt used greedy decoding + default `enable_thinking` — produced dramatic distribution drift from the original Qwen3.5 baseline (cr 64→17, cw 18→78) because (a) greedy is explicitly contraindicated for Qwen3 per `baseline_crossmodel.py` line 37, and (b) the baseline used sampling with `enable_thinking=False`. The B.1 attribution was computed on correctly-sampled baseline responses, so mismatched B.2 gen params would have invalidated the head rankings for the regenerated content. Killed, corrected, relaunched.

## Results — main sweep (seed offset 1000)

| λ | cells (cr/hr/cw/hw) | accuracy | assertion_rate | hallucination_rate | coverage_mean | assertion_asymmetry |
|---|---|---|---|---|---|---|
| 1.0 | 19 / 2 / 72 / 7  | 0.210 | 0.910 | 0.911 | 0.219 | +0.007 |
| 0.9 | 14 / 3 / 75 / 8  | 0.170 | 0.890 | 0.904 | 0.206 | +0.080 |
| 0.8 | 12 / 3 / 75 / 10 | 0.150 | 0.870 | 0.882 | 0.234 | +0.082 |
| 0.5 | 22 / 1 / 63 / 14 | 0.230 | 0.850 | 0.818 | 0.235 | **−0.138** |

## Read

### Qwen3.5 is a much harder case on chef than Gemma

Baseline λ=1.0: acc=0.21, assert=0.91, cw=72. The model is dramatically more confidently-wrong than Gemma (which had acc=0.66, cw=29 at greedy λ=1.0). Chef is hostile territory for 0.8B — it confidently hallucinates French cuisine facts. This actually makes it a **better harness** for testing Humble intervention, because there's abundant cw cohort to shrink and very little pre-existing hedging to mask effects.

### λ=0.9 and λ=0.8: wrong-direction, consistent with Gemma pattern

At modest damping, the intervention **damages correct-confident behavior**:
- cr: 19 → 14 → 12 (monotonic drop)
- cw: 72 → 75 → 75 (unchanged / slightly up)
- accuracy: 0.21 → 0.17 → 0.15 (monotonic drop)

This is the same failure mode as Gemma — the damped heads appear to carry correct-knowledge signal, not disposition signal. Damping erodes what the model knew without fixing overconfidence.

### λ=0.5: first positive separability signal in the project

At aggressive damping, the pattern **inverts**:
- cr: 12 → 22 (recovers *above* baseline)
- cw: 75 → 63 (−12, meaningful shrinkage of confident-wrong)
- hw: 10 → 14 (+4, some cw migrated to hedged-wrong)
- hallucination_rate: 0.882 → 0.818 (**−6.4pts vs λ=0.8, −9.3pts vs λ=1.0**)
- accuracy: 0.15 → 0.23 (recovers above baseline)
- assertion_rate: 0.87 → 0.85 (continues to drop)
- assertion_asymmetry: +0.082 → −0.138 (**sign flip** — now hedges more on wrong than on right)

**This is what DD v2 is supposed to look like**: disposition metric moves in the target direction (more hedging on wrong answers), content metric is preserved or improved, cw→hw migration indicates the model started expressing uncertainty specifically on the content it was wrong about.

### But — non-monotonic path is a red flag

A clean DD v2 signal should be monotonic: λ=0.9 → small positive effect, λ=0.8 → moderate positive effect, λ=0.5 → large positive effect. Instead we see λ=0.9 and 0.8 *degrading* performance, then λ=0.5 *recovering and exceeding* baseline. Possible explanations:

1. **Regime change at aggressive damping.** The heads carry mixed signal (some content, some disposition). At modest damping, you lose the content contribution before the disposition contribution fully activates. At aggressive damping (λ=0.5), the circuit is disrupted enough that the model falls back to a different (more hedged, more honest) mode. This would be a real effect but one requiring aggressive intervention, not clean head-scalar steering.

2. **Sampling artifact.** cr 19→22 is a 3-sample difference; cw 72→63 is 9 samples. With sampling + n=100, some of this is within RNG variance. λ=0.5 could be a lucky seed. Need replication with a different seed to distinguish.

3. **Model fallback mode.** Heavy attention damping could push the model into a degenerate generation mode where outputs are shorter/safer/more template-like, which happens to score better on the coverage and hedge metrics without representing true disposition shift.

## Replication verdict (2026-04-08)

**Replication failed.** λ=0.5 rerun with seed offset shifted 1000→2000 (same model, same heads, same λ, different per-prompt RNG trajectories):

| Run | cells (cr/hr/cw/hw) | acc | assert | hall |
|---|---|---|---|---|
| Main λ=1.0 (baseline in-run) | 19 / 2 / 72 / 7  | 0.210 | 0.910 | 0.911 |
| Main λ=0.5 (seed 1000+i)     | 22 / 1 / 63 / 14 | 0.230 | 0.850 | 0.818 |
| **Rep λ=0.5 (seed 2000+i)**  | **12 / 1 / 77 / 10** | **0.130** | **0.890** | **0.885** |

Every criterion for the "sampling artifact" branch of the pre-declared decision rule was hit:
- hall 0.885 ≥ 0.87 ✓
- cr 12 ≤ 15 ✓ (actually below main λ=1.0)
- cw 77 ≥ 73 ✓
- hw 10 < 12 ✓

The replication reverted to the same content-damage pattern observed at λ=0.9 and λ=0.8 in the main run. The main λ=0.5 was a lucky draw from RNG, not a real intervention effect.

**Qwen3.5 Humble B.2 is falsified.** Cross-model negative confirmed on both Gemma 4 E2B and Qwen3.5-0.8B for Humble.

## Comparison to Gemma Humble

| Metric | Gemma λ=0.5 | Qwen3.5 λ=0.5 |
|---|---|---|
| hall Δ from λ=1.0 | −0.048 | **−0.093** |
| cr Δ from λ=1.0 | −6 | **+3** |
| cw Δ from λ=1.0 | +4 | **−9** |
| assertion Δ | −0.02 | −0.06 |
| coverage Δ | −0.003 | +0.017 |

Every metric moves further in the desired direction on Qwen3.5 than on Gemma. This is consistent with the hypothesis that Qwen3.5's L23 head cluster encodes a more localized/separable signal than Gemma's distributed heads. Whether this survives replication is the open question.

## Implications for the core claim

`CORE_CLAIM_ddv2.md` requires the dual criterion: (a) disposition moves in target direction, (b) content preserved. On Qwen3.5 Humble, **neither holds under replication**. The initial main-run λ=0.5 appeared to meet both but was a sampling artifact — the intervention's effect is RNG-dominated rather than signal-dominated, which means the intervention is producing no reliable causal shift at all.

**Cross-model verdict:** inference-time attention-head tempering via magnitude attribution + top-K damping on `o_proj` input **is falsified as a DD v2 method for the Humble disposition** on both tested architectures:
- **Gemma 4 E2B** (dense, 35 attn layers, distributed signal max d=0.60): frozen disposition, small content drift
- **Qwen3.5-0.8B** (hybrid DeltaNet + 6 attn layers, L23 cluster max d=0.56): non-monotonic noise, no stable effect

Two architectures. Two cohort-distribution shapes. Same null. The method does not produce reliable Humble separability on either.

## What remains open

This finding falsifies a specific configuration, not the broader claim. Still untested:

1. **Directional attribution.** All B.1 runs used L2 norms of per-head activations. Heads with cohort-specific *directions* at similar magnitudes are invisible to this. Cohort-mean vectors + cosine distance is a ~30s re-run on existing data.
2. **Different intervention locations.** We damped `o_proj` *input* (effectively the attention output before projection). Alternatives: `v_proj` output, attention weights, MLP down_proj, residual stream after specific layers.
3. **Other dispositions on Qwen3.5.** Humble is one of seven. Could run Adversarial-self B.1+B.2 on Qwen3.5 using existing baseline if we score a rubric pass first.
4. **Larger K.** Top-5 is arbitrary. Distributed signals may need top-20 or full-layer damping.
5. **Different base models.** Qwen3-0.6B, SmolLM2, Phi-4-mini — different size/architecture/training regimes.

## Recommended next move

Given that the two highest-probability shots (Gemma with strongest-in-cohort localized signal for Adv-self, Qwen3.5 with L23 cluster for Humble) have both failed, the prior on "magnitude-attribution-based head tempering works on Humble on sub-2B models" should be low. Two reasonable paths:

**(A) Cheap additional evidence before closing the inference-time path.**
- Directional re-attribution on both Gemma and Qwen3.5 existing data (~5 min, $0, uses captured norms... actually no, needs re-capture with full activation vectors, ~5 min GPU, $0).
- If directional still null → close inference-time path definitively.

**(B) Accept falsification and pivot.**
- Write a cross-model negative result as the main finding.
- Next intervention class: DPO with explicit dispositional pairs (already attempted in DD v1 for Chef, failed there too — but on a different construction).
- Or: accept DD v2 is not viable at 0.6-2B scale, document the negative, and reframe the research program.

I recommend **(A)** — directional attribution is cheap, it's the single most plausible methodological fix, and running it first prevents the "we never tried X" critique in the writeup. Cost: ~10 min + $0.

## Caveats

1. **Single-seed result.** The entire λ=0.5 effect rests on one RNG trajectory per prompt. Replication is essential before any claim.
2. **Small n per cell.** cr=19 vs 22, cw=72 vs 63 — differences of 3 and 9 samples. Confidence intervals are wide.
3. **Judge stochasticity.** DeepSeek V3.2-exp at temp=0 has ~±2pt noise across calls on the coverage classifier. Not enough to explain the full signal but adds to uncertainty.
4. **Only 3 attention layers active in hook set.** Qwen3.5's hybrid architecture means our hooks fire on L7, L19, L23 — not all 6 attention layers. L3, L11, L15 were not in the top-5. Ablating the full L23 cluster (all 8 heads) might give a cleaner signal than picking 3 from it.
5. **Content damage at λ=0.5 is present but small.** Coverage_mean 0.219 → 0.235 is technically an *improvement* but it's within judge noise. I wouldn't claim coverage improved, only that it didn't drop.

## Files

- `dd-v2/eval/results/path_b2_qwen35_0.8b_humble_summary.json` — 4-λ sweep (seed offset 1000)
- `dd-v2/eval/results/path_b2_qwen35_0.8b_humble_samples.json` — full per-prompt responses + judge outputs
- `dd-v2/eval/logs/path_b2_qwen35.log` — main run log
- `dd-v2/eval/logs/path_b2_qwen35_rep.log` — replication log (in progress)
