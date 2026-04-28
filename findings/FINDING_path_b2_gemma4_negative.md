# Finding: Path B.2 Tempering — Negative Result on Gemma 4 E2B (Humble + Adversarial-self)

**Date:** 2026-04-08
**Status:** Negative — magnitude-attribution + top-K head damping does not shape disposition on Gemma 4 E2B
**Scripts:**
- `dd-v2/eval/path_b2_tempering.py` (Humble)
- `dd-v2/eval/path_b_attr_gemma4_adv.py` (Adv-self B.1)
- `dd-v2/eval/path_b2_gemma4_adv.py` (Adv-self B.2, killed after λ=0.9)
**Raw data:**
- `dd-v2/eval/results/path_b2_tempering_gemma4_summary.json` — full 4-λ Humble sweep
- `dd-v2/eval/results/path_b2_gemma4_adversarial_samples.json` — partial Adv-self (λ=1.0, 0.9)
- `dd-v2/eval/results/path_b_attribution_gemma4_adversarial.json` — Adv-self B.1 ranking
**Depends on:** `FINDING_path_b1_attribution.md`, `CORE_CLAIM_ddv2.md`

## Question

Does damping the top-K heads identified by B.1 attribution causally shift the targeted disposition on Gemma 4 E2B while preserving factual coverage? This is the separability claim at the heart of DD v2.

Two dispositions tested on the same 100 chef prompts: **Humble** (assertion asymmetry / hallucination rate) and **Adversarial-self** (self_verification score).

## Method summary

- Base model: `unsloth/gemma-4-E2B-it`, 35 attention layers × 8 heads = 280 heads total
- Harness: 100 French cuisine prompts from `eval_baseline.json`, gold-checklist coverage as content-stability metric (identical judge across runs: strict DeepSeek V3.2-exp YES/NO per claim, siliconflow/fp8 provider)
- Intervention: `forward_pre_hook` on each target layer's `self_attn.o_proj`, multiplies per-head slice of the input by λ ∈ {1.0, 0.9, 0.8, 0.5}
- Top-5 heads picked by |Cohen d| from B.1 attribution per disposition
- Dual measurement: (1) disposition metric (assertion asymmetry / self_verification score), (2) gold-checklist coverage

## Run 1 — Humble (full sweep)

Top-5 heads from `path_b_attribution_gemma4.json` (all positive d, higher activation on confident-wrong cohort):

| Rank | Layer | Head | Cohen d |
|---|---|---|---|
| 1 | L16 | H5 | +0.599 |
| 2 | L20 | H7 | +0.597 |
| 3 | L3  | H3 | +0.437 |
| 4 | L18 | H5 | +0.431 |
| 5 | L17 | H0 | +0.381 |

Distributed signal across 5 layers, max d modest (0.60).

### Sweep result

| λ | cells (cr/hr/cw/hw) | accuracy | assertion_rate | hallucination_rate | coverage_mean |
|---|---|---|---|---|---|
| 1.0 | 60 / 6 / 29 / 5 | 0.660 | 0.890 | 0.853 | 0.489 |
| 0.9 | 52 / 7 / 36 / 5 | 0.590 | 0.880 | 0.878 | 0.492 |
| 0.8 | 52 / 6 / 35 / 7 | 0.580 | 0.870 | 0.833 | 0.470 |
| 0.5 | 54 / 5 / 33 / 8 | 0.590 | 0.870 | 0.805 | 0.486 |

### Read

- **Assertion rate is frozen** (0.89 → 0.87). The dispositional axis we targeted (confident delivery) does not shift with tempering.
- **Hallucination rate drifts down modestly** (0.853 → 0.805) but this is partly a denominator artifact — cw cell counts are roughly flat (29 → 33).
- **Accuracy drops 7pts from λ=1.0 to λ=0.8** (0.66 → 0.58). Coverage shifts are inside judge stochasticity (±1.5pts), but the 2×2 accuracy count moves reproducibly.
- **No monotonic structure.** λ=0.8 and λ=0.5 are functionally equivalent on every axis except cw cell count.

**Verdict: separability fails.** The top-5 heads carry a small content load and essentially no disposition load. Damping does not move Humble.

## Run 2 — Adversarial-self (partial — λ=1.0, 0.9 only, killed after decision to pivot)

### Cohort definition

Reused `judge_rubric.json` conditions.baseline.scores from Step 16. Positional alignment with baseline items confirmed (both n=100, derived in-order from `eval_baseline.json` rows). Split by `self_verification` rubric axis:
- **advH (high self-verif):** score ≥ 4 → n=11
- **advL (low self-verif):** score ≤ 2 → n=69

Other axes on the same rubric were not usable:
- `pedagogical_framing` (Deliberate): **saturated**, 98/100 high, 1 low — no cohort split possible
- `question_certainty=AMBIGUOUS` (Curious): 0/100 ambiguous — no cohort
- `hedging_appropriateness` (Humble): already covered by Run 1

So Adversarial-self was the **only** disposition besides Humble attributable from existing Gemma chef data.

### B.1 attribution

Top heads by |Cohen d| (sign negative = higher activation on low-self-verif responses):

| Rank | Layer | Head | Cohen d |
|---|---|---|---|
| 1 | L23 | H0 | -0.875 |
| 2 | L11 | H5 | -0.768 |
| 3 | L11 | H0 | -0.732 |
| 4 | L11 | H6 | -0.686 |
| 5 | L15 | H4 | -0.669 |
| 6 | L13 | H0 | -0.620 |
| 7 | L11 | H7 | -0.617 |
| 8 | L17 | H1 | +0.612 |
| 9 | L13 | H4 | -0.602 |
| 10 | L4  | H4 | -0.583 |

**Notable — qualitatively stronger signal than Humble:**
- Max |d| = 0.875 (vs Humble's 0.60)
- 8 heads with |d| ≥ 0.58
- **L11 cluster** — 4 of top-10 heads in layer 11 (H5, H0, H6, H7), suggesting a localized subcircuit
- All top-5 have negative sign → they fire MORE on low-self-verif responses → damping should *increase* self-verification (causal hypothesis)

This was the strongest localized signal we had on Gemma. If magnitude attribution + top-K damping works anywhere on this model, it should work here.

### Sweep result (partial)

Top-5 damped: L23-H0, L11-H5, L11-H0, L11-H6, L15-H4. Disposition metric: single-axis self_verification 1–5 score via DeepSeek V3.2-exp judge (different prompt from the rubric judge; anchors: 1=no self-check, 4=cross-checks a claim OR names a limit, 5=multiple cross-checks + explicit limits).

| λ | sv_mean | sv_high(≥4) | sv_low(≤2) | coverage_mean |
|---|---|---|---|---|
| 1.0 | 3.69 | 82 | 17 | 0.474 |
| 0.9 | 3.58 | 75 | 23 | 0.479 |

**Read:** self_verification moved in the **wrong direction**. Damping heads that were correlated with low self-verif produced a small *decrease* in self_verification, not an increase. Coverage essentially flat.

### Killed after λ=0.9

The λ=0.9 result plus the full Humble sweep gave enough signal to call it. Continuing the sweep would have cost ~1h + ~$5 for data that would strengthen the writeup but not change the decision. Process terminated; λ=0.8 and λ=0.5 not run.

## Combined verdict

Two dispositions tested on Gemma 4 E2B with the strongest correlational signals we could find:

1. **Humble** — distributed weak signal (max d=0.60), full sweep: disposition frozen, content modestly damaged.
2. **Adversarial-self** — localized strong signal with L11 cluster (max |d|=0.875), partial sweep: disposition moves *backward*, content flat.

**The correlational B.1 signal does not survive causal B.2 intervention on Gemma 4 E2B.** The heads with the highest cohort-discriminating activation magnitudes are carrying content distinctions, not disposition distinctions. The disposition signal — if it exists at all — is not localized in the way this method assumes.

## Why this matters for the core claim

`CORE_CLAIM_ddv2.md` states that disposition is a **low-rank, inference-time-steerable axis separable from content**, with each disposition addressable by damping a small set of attention heads. The failure mode "coverage moves with disposition" is the explicit criterion for when the claim fails.

**On Gemma 4 E2B the claim is falsified for Humble.** For Adversarial-self the claim fails more starkly — disposition moves the *wrong way*, which is evidence that the attribution is finding heads that correlate with the cohort split for reasons other than the disposition itself (probably content-category correlation: the kind of questions where Gemma is already self-verifying vs the kind where it isn't).

**But:** one model + two dispositions + one attribution method is not sufficient to kill the general claim. What we've falsified is the narrow version "magnitude attribution + top-K o_proj damping works universally on every disposition on every sub-2B model." The broader hypothesis — that *some* base models have separable disposition circuits addressable by *some* attribution method — remains open.

## What this does NOT rule out

1. **Other base models.** Qwen3.5-0.8B has a structurally different signal: 3 of its top-5 heads cluster in L23, the final attention layer. This is a qualitatively different mechanistic regime (late-layer commitment) not tested by Gemma. Cross-model test is the natural next step. **Status: Qwen3.5-0.8B Humble B.2 launched 2026-04-08 after this finding was drafted.**
2. **Directional attribution.** We measured L2 norms of per-head activations. A head that activates with the same magnitude but different direction on the two cohorts would be invisible to this attribution. Cohort-mean vectors + cosine distance could reveal a separable axis we're missing.
3. **Different intervention location.** We damped `o_proj` input. Could instead damp `v_proj` output, attention weights, or MLP outputs.
4. **Different K or λ schedule.** K=5 may be too small for distributed signals; λ=0.5 may be too aggressive. Neither was systematically varied.
5. **Different prompt harness.** Chef may be a harder benchmark for disposition separability than a task with cleaner cohort splits (e.g., a MCQ benchmark with known correct answers).

## What this DOES imply

- **The simple recipe is not universal.** "Find top-K heads by activation magnitude, damp by λ < 1, expect disposition to move" does not work as a turnkey method across arbitrary base model × disposition combinations.
- **Disposition localization may be model-dependent.** Some architectures/scales may have clean disposition circuits, others may not. This is a testable hypothesis, not a fatal flaw.
- **The writeup becomes more honest, not less valuable.** A paper reporting "method works on models A, B; fails on C, D — here's the pattern" is stronger than a universal claim.

## Decision made after this finding

Pivot to **Qwen3.5-0.8B cross-model replication** of both Humble and Adversarial-self. Cost: ~$12, ~3h GPU. Decision rule:

- **Both Qwen3.5 runs also fail (same pattern)**: inference-time head tempering is falsified as a global DD v2 method. Write up cross-model negative. Consider DPO or alternative intervention classes, or accept DD v2 as falsified at this scale.
- **At least one Qwen3.5 run succeeds cleanly**: method is model-dependent, not universal. Qwen3 family becomes the validated lane (also matches the existing production portfolio — tinman-code/reason/chat are all Qwen3-based). Gemma becomes the negative control in the writeup. Proceed with full 4-disposition Qwen3.5 sweep.

## Caveats on these Gemma runs

1. **Judge stochasticity.** DeepSeek V3.2-exp at temperature 0 is not perfectly deterministic. In-run λ=1.0 drifted from the original Step 16 baseline (cr=60 vs 53, +7 cr cell). Small acc/cov variations across λ should not be over-interpreted.
2. **Cohort sizes.** Humble cw=38 and Adv-self advH=11 are adequate but not large. Cohen's d values are detectable but have non-trivial confidence intervals.
3. **Last-32-token pooling.** Attribution pooled norms over the last 32 response tokens. Different pool lengths were not ablated.
4. **Magnitude attribution only.** L2 norms, not directional alignment. A directional re-run is a cheap follow-up (~30s GPU).
5. **Gemma 4's gated attention.** Unsloth warns "does not support SDPA — switching to fast eager." This affects the kernel but not the attribution math; the o_proj input shape is still `[B, T, n_heads × head_dim]` as expected.

## Files

- `dd-v2/eval/results/path_b2_tempering_gemma4_summary.json` — Humble 4-λ sweep
- `dd-v2/eval/results/path_b2_gemma4_adversarial_samples.json` — Adv-self partial sweep (λ=1.0, 0.9)
- `dd-v2/eval/results/path_b_attribution_gemma4_adversarial.json` — Adv-self B.1 ranking (full 280 heads)
- `dd-v2/eval/logs/path_b2_gemma4.log` — Humble run log
- `dd-v2/eval/logs/path_b2_adv.log` — Adv-self run log
- `dd-v2/eval/logs/path_b_attr_adv.log` — Adv-self attribution log
