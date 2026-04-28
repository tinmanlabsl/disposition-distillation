# DD v2 Core Claim

**Status:** Load-bearing thesis. Do not let experiments collapse this back to "reduce hallucination on chef."
**Date written:** 2026-04-08
**Related:** `STRATEGIC_ground_truth_anchored_calibration.md`, `APPROACH_attention_head_tempering.md`, `FINDING_path_b1_attribution.md`, `FINDING_crossmodel_baseline_disposition.md`

## The claim

Disposition is a **low-rank, inference-time-steerable axis** that is **separable from content** across the full 7-disposition taxonomy. For each disposition, a small set of attention heads can be identified by ground-truth-anchored cohort attribution, and damping or amplifying those heads at inference shapes character **without affecting factual coverage**.

Said differently: character and capability live in separable circuits. We can dial character without retraining, without losing facts.

## The seven dispositions in scope

1. **Eager** — warmth, engagement pitch
2. **Deliberate** — stepwise-ness, pause-before-commit
3. **Adversarial** — self-critique intensity
4. **Curious** — asking vs assuming
5. **Self-Improving** — iteration willingness after feedback
6. **Humble** — uncertainty acknowledgment (what B.2 tests first)
7. **Persistent** — retry-after-failure

**Humble is experiment 1 of 7**, not "the experiment." It's first because it has the cheapest cohort anchor: gold-checklist coverage for correct/wrong × ASSERT/HEDGE judge for delivery. The other six need their own cohort splitters and judges, but the pipeline is the same.

## What we are actually measuring

Every intervention test must measure **both** quantities:

1. **Disposition shift** — does the targeted character dimension actually move? (E.g. confident-wrong rate drops for Humble; warmth score rises for Eager.)
2. **Coverage stability** — does factual coverage on the gold checklist stay flat? This is the separability claim.

**If coverage moves with disposition, the heads carry content, not character, and the claim fails for that disposition.** This is the non-negotiable criterion that separates DD v2 from generic head ablation.

## Method template (repeats per disposition)

1. **Cohort definition.** Split baseline responses into (target-disposition-present) vs (target-disposition-absent) using a judge, crossed with a ground-truth correctness anchor where one exists.
2. **B.1 attribution.** Forward-pass prompt+response, hook `self_attn.o_proj` input per layer, compute per-head L2 norm over last ~32 tokens. Rank heads by Cohen's d between the disposition cohorts.
3. **B.2 tempering.** Forward pre-hook on top-K heads' `o_proj` input slice, multiply by λ. Sweep λ ∈ {1.0, 0.9, 0.8, 0.5}. Regenerate the full benchmark per λ.
4. **Dual measurement.** Track disposition shift AND coverage per λ. Separability is "disposition moves monotonically, coverage stays flat."
5. **Record the head set** as `(layer, head, λ*)` tuples for that disposition on that base model.

## The DD v2 artifact

Not a tempered model. Not a LoRA. The artifact is a mapping:

```
base_model → {disposition: [(layer, head, λ), ...]}
```

Learned **once per base model**, **composed at inference** by stacking forward pre-hooks. No per-specialist retraining. No dataset pipeline per domain. Shipping a "chef specialist" becomes: ship the base model + a 7-line JSON of disposition head sets + a weighting profile (high Humble, high Deliberate, low Eager, etc.).

## How this replaces the old DD v1 plan

CLAUDE.md Phases 1–3 describe a training pipeline: generate ~1,250 prompts per specialist × 4 teacher stages → LoRA train → merge → BitNet quantize. If Path B generalizes across the full taxonomy, **that pipeline is obsolete** for the disposition layer. The training pipeline still applies to *capability* (domain knowledge, code quality, citation grounding) but not to *character*. Character becomes an inference-time config.

## Open questions B.2 and successors must answer

1. **Separability (B.2 Humble):** does coverage stay flat when confident-wrong rate drops?
2. **Modularity (B.2 ×7):** are disposition head sets disjoint or overlapping? Is there a shared "commitment" circuit?
3. **Composability:** can we damp Humble-heads and amplify Curious-heads simultaneously without coverage loss?
4. **Transfer:** do head sets identified on chef prompts work on coding / GDPR / companion prompts, or are they domain-specific?
5. **Cross-model:** is the *count* and *layer distribution* of disposition heads similar across bases (Gemma 4 E2B, Qwen3.5-0.8B, SmolLM2)? Does the method itself generalize?

## Prior art positioning

Head-level inference-time intervention for a *single* behavior is prior art:
- ITI (Li et al. 2023) — truthfulness on TruthfulQA MC
- Band et al. 2025 — verbal uncertainty as a linear feature
- UAH (arXiv 2505.20045, 2025) — uncertainty-aware attention heads (read before publication)
- HACP — hallucination-control heads on LVLMs

**What's not in the prior art (as far as we can tell):**
- A multi-disposition taxonomy treated as a general character-shaping substrate
- Composability of multiple disposition interventions simultaneously
- Ground-truth-anchored cohort split on free-form factual generation (not MC)
- Coverage-stability as the primary success criterion
- A shippable per-base-model head-set mapping as the production artifact

Incrementalism on any one of those points isn't the claim. The claim is the combination.

## Failure modes that collapse the claim back

- "B.2 reduced hallucination on chef" without reporting coverage → reverts to generic head ablation paper.
- Running B.2 only for Humble and declaring DD v2 validated → ignores the taxonomy claim.
- Switching to training (DPO, SFT) after B.2 without exhausting the inference-time taxonomy → reverts to DD v1.
- Reporting novelty against ITI/Band alone → misframes the lane. The lane is taxonomy + composability + content preservation, not "head intervention for confidence."

## What to do next (after B.2 Humble completes)

1. Read the B.2 Humble result with both axes (disposition shift AND coverage).
2. Read UAH (2505.20045) and Band 2025 in full to confirm neither measured coverage-hold-flat across a taxonomy.
3. Draft cohort splitters and judge prompts for the other six dispositions.
4. Pick the next cheapest disposition to attribute (likely Deliberate or Curious — both have detectable verbal signatures).
5. Run B.1 + B.2 for that disposition.
6. After ≥3 dispositions are attributed, measure head-set overlap and compose.
