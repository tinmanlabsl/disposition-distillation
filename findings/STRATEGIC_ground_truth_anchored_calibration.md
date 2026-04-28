# Strategic: Ground-Truth Anchored Calibration — Architectural Principles for DD v2

**Date:** 2026-04-08
**Status:** Active. Supersedes all prior training-target framings in DD v2.

## Context

This document captures two architectural corrections to the DD v2 approach, both raised by the user during the method-validation thread following the α-sweep and PCA falsifications:

1. **Content teachers are situational, not universal.** External content teachers are only net-positive when teacher quality >> base quality on the target content. For a strong base (Gemma 4 E2B), foreign teacher distillation imports a worse distribution masquerading as improvement and produces the drift we have empirically observed (Phase 0 Humaneval regression, Phase 1 reasoning compression, DD v2 chef content collapse). This is not a universal ban on teachers — it is conditional on the strength asymmetry.

2. **Disposition is fixed character, not domain behavior.** Disposition is the same across domains. Chef is the *measurement domain*, not a disposition in itself. What we are trying to instill is the Tinman disposition set (humble, deliberate, curious, self-verifying, etc.) — the same way a human professional has a disposition that travels with them across contexts.

## The hallucination failure mode — why sampling-stability is not a valid confidence proxy

An earlier draft of this plan proposed using sampling stability (agreement across 8 samples at temp 0.9) as a proxy for model confidence, on the reasoning that stable outputs reflect high confidence and varying outputs reflect uncertainty. This is **broken by a critical failure mode**: a model can be stably wrong. Eight consecutive samples can all assert the same hallucinated fact with the same confident delivery. Training a calibration adapter on sampling-stability labels would reward the model for verbalizing confidence on hallucinations — the exact opposite of calibration.

**Ground truth must anchor the calibration label, not the model's own agreement.** For the chef domain, the gold checklist provides this anchor: coverage score = correctness, measured against external reality. The calibration target becomes a 2×2:

|  | Verbal assertion | Verbal hedging / acknowledging limits |
|---|---|---|
| **Content correct** (high coverage) | ✅ appropriately confident | ❌ unnecessarily hedging |
| **Content incorrect** (low coverage) | ❌ confidently hallucinating (worst case) | ✅ appropriately acknowledging limits |

Training pairs are contrasts between samples on the same prompt that land in different cells. The ground-truth anchor ensures the adapter learns "verbalize confidence when content is actually correct; verbalize uncertainty when content is actually wrong" rather than "verbalize whatever matches sampling agreement."

## The calibrated disposition constraint

Disposition is not a blanket style. "Humble" does not mean "always hedge." A model that hedges on every answer is not humble — it is miscalibrated and useless. The disposition must track actual epistemic state:

- Verbalize confidence when the content is actually correct
- Verbalize uncertainty when the content is actually wrong or unknown
- NOT insert uncertainty markers as a stylistic habit
- NOT insert confidence markers as a stylistic habit

Operationally, the disposition adapter must be a **pass-through from true epistemic state to explicit verbal markers**, not a style transformation. The training signal must penalize miscalibration in both directions.

## The "eager to be right" baseline problem

RLHF-tuned models (including Gemma 4 E2B) have typically been rewarded for sounding confident and helpful. This produces a baked-in "eager-to-appear-correct" disposition that predates any DD intervention. Before we can instill calibrated disposition, we need to measure this baseline quantitatively:

**Baseline disposition profile questions:**
- What fraction of Gemma 4 E2B's outputs on chef prompts are confidently delivered?
- What fraction of those confident outputs are actually correct vs hallucinated?
- What is the asymmetry between assertion rate and accuracy?
- Does Gemma assert more readily than its content accuracy would justify?

If the baseline shows high confident-wrong rates (e.g., >30% of wrong answers delivered confidently), the DD intervention is not "add humble disposition" — it is **subtract the miscalibrated eager-confident disposition that RLHF installed**. This is a subtractive intervention with a very different objective than any prior DD attempt.

## Architectural principles (locked in)

1. **Content distribution:** self-supervised (base's own outputs) when base is strong; external teachers only when teacher >> base on target content. For Gemma 4 E2B on general-to-moderate domains, self-supervised is the default.

2. **Disposition target:** ground-truth anchored calibration, not sampling-stability agreement, not teacher-style mimicry.

3. **Intervention preference order:**
   1. Inference-time (no weight modification) — CCA-steering, attention-head tempering. Cannot damage content by construction.
   2. Weight-training with tight content-preservation regularizer — only if inference-time fails.
   3. Foreign-teacher distillation — only for domains where base is weak and teacher is genuinely stronger. Not a DD v2 default.

4. **Domain selection:** chef is the measurement domain because it is non-deterministic, narrow, tangential to base's core competence, and permits disposition to express without tripping a hard right/wrong gradient. No domain switch for training; chef throughout.

5. **Baseline before intervention:** quantitative measurement of pre-intervention disposition profile is mandatory. We cannot measure intervention effect without a baseline.

## What this kills from the prior plan

- Sampling-stability-proxy calibration training (would amplify hallucination confidence)
- VGSD as a thesis test (VGSD is content-restoration, not disposition, and it is orthogonal to the actual goal — VGSD feasibility data is kept as a null baseline but the training step will not be run)
- Further measurements on `chef_dpo` as a research artifact (it is entangled with foreign teacher content and cannot cleanly test disposition; see `ARCHIVE_chef_dpo_lessons.md`)

## What survives

- Chef as the measurement domain
- VGSD feasibility data (repurposed as the ground-truth corpus for Phase 0 baseline measurement — coverage scores already computed)
- α-sweep and PCA findings (both remain valid — they falsify per-token mixing and per-vector steering respectively)
- The thesis that disposition is learnable and separable from content, tested under the corrected framing

## Next steps

1. **Phase 0:** Baseline disposition measurement using VGSD data + verbal-marker classification. Produces quantitative profile of Gemma 4 E2B's pre-intervention disposition. See `APPROACH_baseline_disposition_measurement.md`.
2. **Path decision:** Based on Phase 0 results, choose between attention-head tempering (Path B, mechanistic, inference-time, no training) and calibration-DPO LoRA (Path A, training, ground-truth-anchored pairs). See `APPROACH_attention_head_tempering.md` and `APPROACH_calibration_dpo.md`.
3. **User sign-off gates:** baseline results → path choice → intervention launch. No training or intervention without explicit approval.
