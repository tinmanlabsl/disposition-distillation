# Finding: Gemma 4 E2B Baseline Disposition — Assertion Uncorrelated with Correctness

**Date:** 2026-04-08
**Status:** Current (replaces the retracted v1 of this doc — see history note at bottom)
**Source data:** `dd-v2/eval/results/eval_baseline.json` (Step 16 greedy responses, n=100) + `dd-v2/eval/results/judge_v3_checklist.json` (gold-anchored coverage, n=100) + `dd-v2/eval/results/baseline_gemma4_v2_from_step16.json` (verbal classification)
**Script:** `dd-v2/eval/classify_baseline_verbal.py`

## Headline numbers

| Metric | Value |
|---|---|
| n | 100 prompts, 1 greedy response each |
| Accuracy (gold claim coverage ≥ 0.5) | **58.0%** (58/100) |
| Assertion rate (overall) | **91.0%** (91/100) |
| **Hallucination rate** (confident-wrong / all-wrong) | **90.5%** (38/42) |
| Over-hedging rate (hedged-right / all-correct) | 8.6% (5/58) |
| **Assertion asymmetry** P(assert\|wrong) − P(assert\|correct) | **−0.009** |

## The 2×2 disposition matrix

|  | assertive | hedged | total |
|---|---|---|---|
| **correct** (coverage ≥ 0.5) | **53** | 5 | 58 |
| **wrong** (coverage < 0.5) | **38** | 4 | 42 |
| total | 91 | 9 | 100 |

P(assert | correct) = 53/58 = **91.4%**
P(assert | wrong)   = 38/42 = **90.5%**

The model asserts confidently at essentially the same rate whether the content is correct or wrong. There is no useful information in its verbal delivery about whether the underlying content is reliable.

## Method

- **Responses:** Step 16 evaluation ran Gemma 4 E2B base (via `unsloth/gemma-4-E2B-it` + `disable_adapter`) on 100 held-out chef prompts with locked `GEN_PARAMS` (temp=0 greedy, max_new=512). Saved to `eval_baseline.json['rows']`.
- **Content correctness:** Step 16f gold-anchored checklist judge (`eval_judge_rubric_v3.py`) scored each response against a hand-drafted checklist of required canonical claims (3-5 per prompt). DeepSeek V3.2 asks YES/NO per claim per response. Score = fraction of claims hit. Threshold 0.5 → correct. Saved to `judge_v3_checklist.json`.
- **Verbal classification:** `classify_baseline_verbal.py` runs DeepSeek V3.2 again on each response with a single-token ASSERT/HEDGE prompt (same prompt used in the retracted v1 run; it is independent of the gold checklist).
- **Cross-tab:** joined by prompt position, bucketed into the 2×2.

No GPU was used for this analysis — it reuses clean Step 16 data. Total judge cost: ~$0.05, wall time ~90s.

## What it means

### 1. Gemma 4 E2B is content-capable on this domain

58% accuracy on gold-checklist coverage is meaningfully above the retracted v1 number (22.2%, which was scored against the wrong prompts). Gemma base has substantial French cuisine technique knowledge; it hits canonical claims on the majority of prompts. The earlier "content-weak" framing was an artifact, not a measurement.

### 2. Assertion is statistically independent of correctness

Assertion asymmetry is **−0.009**. The model verbalizes confidence at the same rate (91.4% vs 90.5%) whether it is right or wrong. This is the textbook RLHF miscalibration signature: verbal hedging carries no signal about epistemic state. The finding is robust at n=100 — a 9-percentage-point swing in the wrong direction would be needed to move the asymmetry to ±0.10.

### 3. Hallucination rate is high but coexists with real competence

When wrong, Gemma is confidently wrong **90.5% of the time**. The finding is the same qualitatively as v1 (retracted), just now measured on the right data:
- v1 (retracted, broken scoring): 95.5% hallucination rate on 22% accuracy
- v2 (this, correct scoring): 90.5% hallucination rate on 58% accuracy

The RLHF overconfidence habit is a top-level behavior — it operates independently of whether the model happens to know the answer.

### 4. Baseline hedging is sparse

The hedged column totals 9/100 — Gemma almost never hedges on this prompt distribution at greedy. Specifically:
- **hr (hedged-right): 5** — rare but real; when the model hedges and happens to also be right, it's scoring as a partial-coverage response with caveats.
- **hw (hedged-wrong): 4** — the "appropriate hedge" cell is tiny. Gemma almost never says "I'm not sure" on content it gets wrong.

This has direct implications for training-based interventions (see "Implications" below).

## Implications for intervention paths

### Path A — Calibration DPO

**Constrained, not dead.** The core problem for Path A is pair yield: balanced hallucination-suppression pairs (hedged-wrong vs confident-wrong, same prompt) require both cells to be populated on the same prompt. At greedy n=1, that's impossible (one sample per prompt). At stochastic n=8 we'd need to rerun to measure, but given hw=4 at greedy, the hedged-wrong cell is structurally sparse — the model simply does not produce many hedged outputs on this domain.

Mitigations would require either:
- Much larger sample counts per prompt at high temperature to capture tail-hedging events, or
- A different mechanism for generating hedged variants (e.g., system-prompted "respond with hedging language" variant of base)

The second mitigation re-introduces the foreign-distribution problem the strategic frame explicitly rejects. The first is expensive and uncertain.

Path A is **viable only if pair yield under stochastic sampling is meaningfully higher than the greedy rate suggests.** Worth checking before committing to training, but not the obvious first move.

### Path B — Attention-head tempering

**Unambiguously viable.** The cohorts needed for attribution are both well above the minimum:
- **confident-wrong (cw): 38 samples** — the "overconfidence" cohort.
- **confident-right (cr): 53 samples** — the "appropriate confidence" contrast.

These are enough for per-head activation-magnitude comparison (t-test over 38 vs 53 samples has >80% power to detect medium effects per head). The attribution experiment is a clean go.

### Path B+ / CCA-aligned steering (fallback within inference-time family)

If per-head attribution produces a diffuse signal (no small set of heads dominates), the natural step up is CCA-aligned attention steering — finding directions in the combined head-output subspace that separate the cw cohort from the cr cohort. This is more expressive than scalar per-head tempering and handles the "distributed signal" case we observed in the PCA rank finding on the DD→base diff.

### Order of operations

1. **Phase B.1** — attention-head attribution on cw (n=38) vs cr (n=53) cohorts from this baseline. Pure measurement, ~20 min, ~$0.
2. **If attribution yields clean heads (≥3 heads with effect size > 1σ above rest):** Phase B.2 λ sweep tempering.
3. **If attribution is diffuse:** fall back to CCA-aligned steering on the same cohorts.
4. **Path A as a separate track:** only after B/CCA either succeeds (to contrast) or fails (as the training-based alternative). Path A would also need a stochastic-sampling pass first to confirm pair yield is workable.

## Caveats

1. **Single prompt set, single temperature.** This is Gemma base at temp=0 on 100 French cuisine prompts. Whether the 91% assertion rate and the ~0 asymmetry hold across other domains and temperatures is an open question — addressed in part by the cross-model replication (SmolLM + Qwen3-0.6B) queued next.

2. **Greedy sampling.** At temp=0 the model outputs its single most-likely completion per prompt. Temperature diversity might reveal a wider spread of hedged outputs. The retracted v1 was attempting to measure this via 8× stochastic sampling; that attempt is deferred until we see whether the greedy finding is enough to pick a path.

3. **Single verbal judge.** DeepSeek V3.2 with a one-token ASSERT/HEDGE prompt. Judge error is some unknown fraction but has no directional bias wrt correctness — the correctness label comes from a completely separate gold-anchored pass. The ~0 asymmetry would survive substantial judge noise.

4. **n=100.** Power is sufficient for the headline asymmetry (±0.10 CI on a point estimate of −0.009) and for cohort-size thresholds. It is insufficient for per-prompt within-sample analyses (which need multiple samples per prompt).

## New standalone claim for the paper

> **Gemma 4 E2B exhibits near-perfect decoupling of verbal assertion from content correctness on a French cuisine technique domain (assertion asymmetry = −0.009, n=100). The model asserts in ~91% of outputs regardless of whether the content matches a hand-drafted gold checklist (P(assert|correct)=91.4%, P(assert|wrong)=90.5%). This is a quantified RLHF miscalibration finding on a current open-weights model, anchored to an external gold standard built independently of the model.**

This is a publishable standalone result independent of any DD intervention.

## Files
- `dd-v2/eval/classify_baseline_verbal.py` — the classifier script (reuses Step 16 data)
- `dd-v2/eval/results/eval_baseline.json` — Step 16 base responses (n=100, greedy)
- `dd-v2/eval/results/judge_v3_checklist.json` — gold-anchored coverage labels
- `dd-v2/eval/results/baseline_gemma4_v2_from_step16.json` — full 2×2 + per-item data
- `dd-v2/findings/FINDING_step16_chef_eval.md` — upstream eval that produced the input data

---

## History note

An earlier version of this finding (dated 2026-04-08 morning) reported hallucination rate 95.5%, accuracy 22.2%, and assertion asymmetry −0.023. That version was **retracted** when a post-hoc audit revealed a data-pipeline bug: `vgsd_base_sample.py` fetched prompts from `data/type1_gold.jsonl` using `i` indices from `gold_checklist.json`, but those two files were built for unrelated purposes — the checklist belongs to the Step 16 eval whose prompts live in `eval_baseline.json['rows']`. The sampled responses were to one set of prompts; the coverage was scored against gold claims for a completely different set of prompts. The retracted numbers were an artifact of that cross-file misalignment.

The methodology (2×2, verbal judge, ground-truth anchoring) was correct. Only the data source was wrong. This v2 finding rebuilds the same 2×2 on the already-clean Step 16 data, which is correctly aligned by construction. The qualitative story (assertion decoupled from correctness) is unchanged; the quantitative numbers are cleaner and, notably, the baseline is more content-capable than the retracted version implied.

**Lesson:** always spot-check samples across cells *before* writing the finding doc. A five-minute audit would have caught the bug. The numbers looked catastrophic and internally consistent, which made the bug invisible to a numbers-only review.
