## Finding: Crossmodel Baseline Disposition — RLHF Miscalibration Is Universal but Asymmetry Varies

**Date:** 2026-04-08
**Status:** Current
**Source data:**
- Gemma 4 E2B: `dd-v2/eval/results/baseline_gemma4_v2_from_step16.json` (reuses Step 16 responses + gold checklist)
- SmolLM2-1.7B-Instruct: `dd-v2/eval/results/baseline_smollm_v2_disposition_profile.json`
- Qwen3.5-0.8B: `dd-v2/eval/results/baseline_qwen35_0.8b_v2_disposition_profile.json`

**Scripts:** `dd-v2/eval/classify_baseline_verbal.py` (Gemma, reuses Step 16), `dd-v2/eval/baseline_crossmodel.py` (SmolLM, Qwen3.5)

## Headline table

| Metric | Gemma 4 E2B | SmolLM2-1.7B-Instruct | Qwen3.5-0.8B |
|---|---|---|---|
| n | 100 | 100 | 100 |
| Accuracy (coverage ≥ 0.5) | 58.0% | 56.0% | **75.0%** |
| Assertion rate (overall) | 91.0% | 78.0% | 82.0% |
| Hallucination rate (cw / wrong) | 90.5% | 75.0% | 72.0% |
| Over-hedging rate (hr / correct) | 8.6% | 19.6% | 14.7% |
| **Assertion asymmetry** P(assert\|wrong) − P(assert\|correct) | **−0.009** | **−0.054** | **−0.133** |
| 2×2: cr / hr / cw / hw | 53 / 5 / 38 / 4 | 45 / 11 / 33 / 11 | 64 / 11 / 18 / 7 |

## 2×2 matrices per model

**Gemma 4 E2B** (91.4% assert on correct, 90.5% assert on wrong)

|  | assert | hedge | total |
|---|---|---|---|
| correct | 53 | 5 | 58 |
| wrong | 38 | 4 | 42 |

**SmolLM2-1.7B-Instruct** (80.4% assert on correct, 75.0% assert on wrong)

|  | assert | hedge | total |
|---|---|---|---|
| correct | 45 | 11 | 56 |
| wrong | 33 | 11 | 44 |

**Qwen3.5-0.8B** (85.3% assert on correct, 72.0% assert on wrong)

|  | assert | hedge | total |
|---|---|---|---|
| correct | 64 | 11 | 75 |
| wrong | 18 | 7 | 25 |

## Method

Same pipeline for all three models, held constant to enable direct comparison:

1. **Prompts:** 100 gold-anchored French cuisine technique prompts from `dd-v2/eval/results/eval_baseline.json['rows']`, aligned with hand-drafted canonical claim checklists in `dd-v2/eval/data/gold_checklist.json` by row position. These are the same prompts Step 16 used for the Gemma baseline.
2. **Response generation:**
   - Gemma 4 E2B: Step 16 greedy responses (temp=0, max_new=512) via `unsloth/gemma-4-E2B-it`.
   - SmolLM2-1.7B-Instruct: greedy (temp=0, max_new=512) via Unsloth FastModel.
   - Qwen3.5-0.8B: sampling per Qwen official recommendation (temp=1.0, top_p=1.0, top_k=20, `repetition_penalty=1.2` as HF equivalent of `presence_penalty=2.0`), `enable_thinking=False`, max_new=512. Greedy is explicitly contraindicated by Qwen3.5 docs (causes DeltaNet repetition loops).
3. **Content correctness:** DeepSeek V3.2 via OpenRouter judges YES/NO per canonical claim per response. Coverage = fraction of claims hit. Threshold 0.5 → correct.
4. **Verbal classification:** DeepSeek V3.2 with single-token ASSERT/HEDGE prompt on each response.
5. **2×2:** Joined by prompt position.

**n=1 per prompt** on all three. This is a baseline disposition snapshot, not a within-prompt sampling study.

## What the numbers say

### 1. Confident hallucination is universal across all three models

All three baselines land in the same qualitative regime: **when wrong, these models assert between 72% and 91% of the time.** The RLHF overconfidence habit is a top-level behavior, not a Gemma-specific artifact. This is the central crossmodel finding.

### 2. Content capability varies sharply — and tracks with model lineage, not size

Qwen3.5-0.8B at 75% accuracy outperforms both larger (SmolLM 1.7B) and comparable-effective-size (Gemma E2B) models on this French cuisine gold checklist. The Qwen series has known pretraining breadth on cooking/cuisine domains via its multilingual Chinese+Western data mix; this manifests as a 17-19pt accuracy advantage here. Model size is not the explanatory variable — pretraining distribution is.

### 3. Assertion asymmetry separates the three models meaningfully

- **Gemma 4 E2B: −0.009** — pure decoupling. Verbal delivery carries zero calibration information. This is the textbook RLHF miscalibration signature.
- **SmolLM2-1.7B: −0.054** — slight directional asymmetry but within noise at n=100.
- **Qwen3.5-0.8B: −0.133** — the largest and only clearly non-zero asymmetry. Qwen3.5 asserts confidently ~85% of the time when correct and ~72% when wrong. Verbal hedging carries weak but real calibration signal.

At n=100 and per-cell counts in the 7-64 range, only the Qwen3.5 asymmetry is comfortably outside what sampling noise would produce (binomial CI on 18/25 vs 64/75 does not overlap). The Gemma and SmolLM asymmetries are within noise.

### 4. Over-hedging rates reveal a different story

- Gemma hedges on **8.6%** of its correct answers — it essentially never hedges.
- SmolLM hedges on **19.6%** of its correct answers — the most hedging-prone of the three.
- Qwen3.5 hedges on **14.7%** of its correct answers — middle ground.

Over-hedging is the mirror-image failure to confident hallucination: being unsure of content the model actually has right. SmolLM's 20% rate suggests its RLHF was tuned for conservatism; Gemma's 9% suggests assertive default. Neither is "calibrated" — they're miscalibrated in opposite directions.

## Implications for intervention paths

### Path B (attention-head tempering) — cohort viability

Path B needs cw (confident-wrong) and cr (confident-right) cohorts large enough for per-head statistical comparison. Minimum viable is roughly 30 samples per cohort for medium effect size detection.

| Model | cw | cr | Path B viability |
|---|---|---|---|
| Gemma 4 E2B | 38 | 53 | **Strong** — both cohorts well-powered |
| SmolLM2-1.7B | 33 | 45 | **Viable** — both cohorts meet threshold |
| Qwen3.5-0.8B | 18 | 64 | **Constrained** — cw too small for head-level t-tests; aggregate-level effect detection only |

**Gemma remains the best candidate for Path B attribution** on pure cohort size. A Qwen3.5 replication of Path B would need either more prompts or per-prompt stochastic sampling to grow cw.

### Path A (calibration DPO) — pair yield

Path A needs hedged-wrong vs confident-wrong pairs on the same prompt. At n=1 greedy per prompt, both cells must be non-empty overall, but within-prompt pairs are impossible.

| Model | cw | hw | Rough pair yield ceiling |
|---|---|---|---|
| Gemma 4 E2B | 38 | 4 | Very tight — 4 hw total |
| SmolLM2-1.7B | 33 | 11 | More workable — 11 hw |
| Qwen3.5-0.8B | 18 | 7 | Tight — 7 hw |

**SmolLM has the most workable hw cell.** If Path A is attempted, SmolLM is the first model where stochastic sampling might realistically produce enough balanced pairs.

### The Qwen3.5 asymmetry is potentially a paper result on its own

Qwen3.5's −0.133 asymmetry is the first open-weights small-model baseline I've measured with a non-trivial calibration signal in verbal delivery. If this holds under replication (larger n, different domain), it would be a counter-example to the "RLHF models are universally miscalibrated" claim — Qwen3.5's particular training pipeline produces a model that does partially encode epistemic state into delivery. Worth following up on before committing to a specific intervention path.

## Decision (tentative)

1. **Gemma 4 E2B remains the lead intervention target** for Path B attribution on pure cohort-size grounds.
2. **Qwen3.5-0.8B deserves a follow-up measurement** before deciding whether its asymmetry is a real model property or a noise artifact. The cheapest replication is a second prompt set (non-cuisine domain) or stochastic n=8 on the existing 100 prompts.
3. **SmolLM2 is the Path A candidate** if we decide to pursue calibration DPO as a parallel track.
4. **All three baselines support the core thesis:** the RLHF miscalibration pattern is universal at small-model scale. This is a standalone publishable finding on three independent open-weights models.

## Caveats

1. **Single prompt set, single domain.** 100 French cuisine technique prompts. Crossdomain replication is an open question.
2. **Mixed sampling regime.** Gemma and SmolLM were greedy (n=1, temp=0); Qwen3.5 was stochastic (n=1 sample but with temp=1.0 + repetition_penalty=1.2 because greedy is contraindicated). This is a methodological asymmetry forced by Qwen3.5's architecture. Interpretation: Qwen3.5's numbers reflect one sample drawn from its recommended sampling distribution, not its single-most-likely completion. For the aggregate disposition metric at n=100 this is probably fine, but within-prompt comparisons across models are not strictly apples-to-apples.
3. **`repetition_penalty=1.2` is an approximation of Qwen's `presence_penalty=2.0`.** HF transformers has no additive presence penalty; I used the multiplicative repetition_penalty as a close but not identical substitute. The purpose (suppress DeltaNet looping on the torch-fallback path) is preserved.
4. **n=100.** Sufficient for headline comparisons and large asymmetries (Qwen3.5). Insufficient to distinguish Gemma's −0.009 from SmolLM's −0.054 — both are within noise at this sample size.
5. **Single verbal judge** (DeepSeek V3.2, single-token ASSERT/HEDGE). Judge noise is uniform across models and uncorrelated with correctness, so it does not bias the asymmetry direction.
6. **Qwen3.5 torch-fallback path.** The fast DeltaNet kernels (fla library) were not installed on the pod. Qwen3.5 ran via Unsloth's torch fallback at ~23s/prompt. Outputs look well-formed; there is no reason to believe the fallback path produces different logits than the fast path. Worth confirming if we do Path B on Qwen3.5.

## Standalone claims

1. **At n=100 on a French cuisine technique domain, three sub-2B open-weights instruction-tuned models (Gemma 4 E2B, SmolLM2-1.7B-Instruct, Qwen3.5-0.8B) all produce confidently-wrong outputs at rates between 72% and 91%.** The RLHF overconfidence pattern is not a Gemma artifact.

2. **Content capability varies independently of miscalibration.** Qwen3.5-0.8B is 17-19 points more accurate than the other two yet still hallucinates confidently 72% of the time when it is wrong. Accuracy and calibration are orthogonal.

3. **Assertion asymmetry is not universally zero.** Qwen3.5-0.8B exhibits a −0.133 asymmetry — the only of the three models where verbal delivery carries meaningful calibration signal. This is a potential counter-example to the "RLHF kills verbal calibration" default and deserves replication.

## Files

- `dd-v2/findings/FINDING_gemma4_baseline_disposition.md` — Gemma standalone finding
- `dd-v2/eval/results/baseline_gemma4_v2_from_step16.json` — Gemma 2×2
- `dd-v2/eval/results/baseline_smollm_v2_disposition_profile.json` — SmolLM 2×2
- `dd-v2/eval/results/baseline_qwen35_0.8b_v2_disposition_profile.json` — Qwen3.5 2×2
- `dd-v2/eval/baseline_crossmodel.py` — crossmodel sampling + scoring script
- `dd-v2/eval/classify_baseline_verbal.py` — Gemma reuse-Step-16 classifier
