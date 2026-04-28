# Approach: Attention-Head Tempering (Path B)

> **⚠️ STATUS CORRECTION 2026-04-08:** This approach was briefly promoted to "the only viable path" in `FINDING_gemma4_baseline_disposition.md`. That promotion has been **retracted** — it was based on confident-wrong / confident-right cohort sizes computed on a corrupted 2×2 (wrong dataset indexing in `vgsd_base_sample.py`; prompts came from one dataset, gold claims from another). Cohort sizes will be recomputed after the Phase 0 baseline is rerun with the fix. Path B remains a candidate, not a confirmed direction.

**Date:** 2026-04-08
**Status:** Documented. Launch gated on (corrected) Phase 0 baseline.
**Depends on:** `STRATEGIC_ground_truth_anchored_calibration.md`, `APPROACH_baseline_disposition_measurement.md`

## Core hypothesis

Overconfident delivery on hallucinated content is a behavior shaped primarily by a subset of attention heads, not by MLP knowledge storage. If we can identify the specific heads that fire more strongly on confidently-wrong generations than on confidently-right generations, we can temper those heads at inference time — multiplying their outputs by a scalar λ < 1 — and measure whether hallucination drops without damaging correct-content delivery.

This is a **mechanistic interpretability intervention**, not a training method. It satisfies the DD v2 architectural principle perfectly: no weight modification, no foreign content, no distribution drift, no training data required. The intervention is falsifiable per-head.

## Why attention, not MLPs

- **Geva et al. 2021 (Transformer Feed-Forward Layers Are Key-Value Memories):** MLP layers store factual knowledge as key-value associations. Modifying MLPs damages content.
- **Elhage, Olsson et al. (Anthropic circuits work):** attention heads shape response style, instruction-following behavior, and delivery patterns.
- **RLHF mechanistically:** preference optimization over response shapes affects attention much more than MLPs, because preferences are over styles not over facts.
- **Our own empirical finding:** attn-only LoRA preserves coding knowledge on Qwen3 while all-modules LoRA destroys it. The mechanistic explanation was always "disposition lives in attention, content lives in MLPs" — we just did not articulate it until the strategic reframe.

This gives strong priors that "eager-confident delivery on wrong content" is an attention-head behavior, not a MLP knowledge artifact.

## Phase B.1 — attribution (finding the overconfidence heads)

**Setup:** use the 800-sample VGSD corpus, now labeled by the Phase 0 baseline as confident-right / confident-wrong / hedged-right / hedged-wrong.

**Attribution method:** for each attention head h in the Gemma 4 E2B base model, compute the mean activation magnitude of head h on:
- confident-wrong samples (C_wrong)
- confident-right samples (C_right)
- hedged-wrong samples (H_wrong)

The "overconfidence score" for head h is:

```
overconfidence_score(h) = mean_activation(h | confident-wrong) - mean_activation(h | confident-right)
```

A positive and large score means head h fires more strongly when the model is about to hallucinate confidently. These are the candidates.

**Implementation:** forward pass only (no gradients, no training). Hook each attention head, extract output norms at the last few generated tokens (where confidence/commitment is expressed), average across the confident-wrong and confident-right cohorts, rank heads by score.

**Cost:** 800 forward passes × ~1.5s each on 4090 = ~20 min. No API calls. $0.11.

**Output:** ranked list of the top-k overconfidence heads + attribution confidence intervals.

## Phase B.2 — tempering experiment

**Setup:** pick the top-5 overconfidence heads. For each tempering factor λ ∈ {1.0, 0.8, 0.5, 0.2, 0.0}, regenerate the 100 gold-checklist prompts with those heads multiplied by λ.

**Tempering mechanism:** at inference time, hook the attention output of the selected heads and multiply by λ. This is a lightweight forward-hook modification, no training, no persistent state change. Can be toggled per-prompt.

**Measurement per λ:**
1. Content coverage (gold checklist score) — did we preserve what base knew?
2. Hallucination rate (confident-wrong fraction) — did we reduce misplaced confidence?
3. Over-hedging rate (hedged-right fraction) — did we accidentally make correct content sound uncertain?
4. Overall disposition calibration score

**Success signature:**
- Hallucination rate drops substantially
- Confident-right rate preserved or only mildly reduced
- Content coverage preserved (≥ base)
- Monotonic effect with λ (smaller λ = more tempering = more reduction in hallucination, up to a point)

**Failure signature:**
- Content coverage drops with λ → heads carry content, not just disposition
- Over-hedging rate grows with λ → we are making everything sound uncertain, not specifically wrong content
- No effect with λ → heads are not causally connected to the confident-wrong behavior
- Non-monotonic effect → the intervention is noise

## Phase B.3 — generalization check

**Setup:** apply the selected heads + chosen λ to a NEW set of chef prompts (held-out from Phase 0 and Phase B.2). Verify the tempering effect generalizes, not just fits the attribution set.

**Extra (if time):** apply the same tempering to a non-chef domain (e.g., commonsense QA) to check whether the overconfidence heads are domain-general or chef-specific. Domain-general = stronger thesis validation.

## Cost total

- B.1 attribution: ~$0.11 (GPU only)
- B.2 sweep: 100 prompts × 5 λ values × 400 tokens = 200K tokens of generation; ~40 min on 4090 = ~$0.25; judge scoring ~$1.50
- B.3 generalization: ~$0.50
- **Total: ~$2.50, ~90 min wall time.**

## Why this is the strongest experiment we have conceived

- Satisfies the architectural principle (no weight change, no foreign content)
- Directly tests the "disposition lives in attention" mechanistic hypothesis
- Produces a falsifiable per-head result
- Cheapest experiment on the table
- If it works, it shows disposition is not only separable from content but localizable to specific circuits — a much stronger result than any trained LoRA
- If it fails, we still get an attribution map that tells us where the behavior is NOT localized

## What this does not test

- Whether the tempering effect is trainable into weights (that would be a subsequent experiment)
- Whether tempering generalizes across diverse domains (Phase B.3 samples one)
- Whether the effect holds across different base models (Gemma 4 E2B only)

## Dependencies on Phase 0

Phase B.1 attribution requires labeled cohorts (confident-wrong, confident-right, etc.) which come from Phase 0 output. If Phase 0 shows too few confident-wrong samples (< 20 across all prompts), attribution will be noisy and Path A becomes more attractive.

## Comparison to Path A (calibration DPO)

| Aspect | Path A (DPO) | Path B (tempering) |
|---|---|---|
| Weight modification | Yes, new LoRA | No |
| Training required | Yes (~2h) | No |
| Data required | ~80 DPO pairs | Existing 800 samples |
| Content risk | Possible drift | Near-zero by construction |
| Interpretability | Low (opaque LoRA) | High (per-head attribution) |
| Reversibility | Requires adapter toggle | Per-prompt scalar |
| Cost | ~$3 | ~$2.50 |

Path B is preferred unless attribution produces no clean signal.

## Implementation notes

- Use HuggingFace forward hooks on `model.model.layers[L].self_attn.o_proj` for each layer
- Identify the head index within the projection output via slicing (head_dim × n_heads)
- Apply tempering as `out[:, :, h_start:h_end] *= λ` before returning
- Toggle via a global config dict checked in the hook
