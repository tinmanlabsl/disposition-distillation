# DD v2 Chef — Step 16 Summary

**Date:** 2026-04-08
**Status:** Complete. Decision pending: v2.1 retrain vs RFT pivot.
**Goal:** Method validation, not shipping. Test whether DD v2 imitation pipeline installs adaptive disposition into sub-2B models.

---

## Tests run

| step | what | output file |
|---|---|---|
| 16 | 5-condition ablation eval (n=100) on ChefVerifier | `eval_baseline.json`, `eval_sft_only.json`, `eval_full_ddv2.json`, `eval_single_teacher_sft.json`, `eval_multi_teacher_sft.json` |
| 16b | DeepSeek V3.2 head-to-head judge, full_ddv2 vs baseline | `judge_head2head.json` |
| 16c | DeepSeek V3.2 5-axis rubric judge | `judge_rubric.json` |
| 16d | Inference-time mitigation: full_ddv2 + register-grounding system prompt | `eval_full_ddv2_sysprompt.json`, `judge_rubric_sysprompt.json` |
| 16e | Two-pass response-blind rubric judge (v2) | `judge_rubric_v2.json`, `cert_labels.json` |
| 16f | Gold-anchored binary checklist judge (v3, deterministic) | `judge_v3_checklist.json`, `data/gold_checklist.json` |

All 5 conditions used identical locked GEN_PARAMS (temp=0.0, max_new_tokens=512, do_sample=False, BATCH=32, Gemma 4 E2B base, LoRA adapters via Unsloth FastModel). All judge calls used locked JUDGE_PARAMS (DeepSeek V3.2, temp=0.0).

## Headline numbers

### ChefVerifier (Step 16)

| condition | avg_score | pass_rate |
|---|---|---|
| baseline | 0.279 | 0.02 |
| single_teacher_sft | 0.291 | 0.06 |
| multi_teacher_sft | 0.285 | 0.02 |
| sft_only | 0.270 | 0.04 |
| full_ddv2 | 0.241 | 0.05 |
| full_ddv2 + sysprompt | 0.282 | 0.14 |

All conditions cluster in 0.24–0.29. Verifier known to be coarse and to penalize stylistic shifts (LaTeX formatting, pedagogical framing).

### DeepSeek V3.2 head-to-head (16b)

- full_ddv2 win-rate: **45%**, baseline: **52%**, ties: 3%

### Rubric judge v1 (16c)

| condition | factual | hedging | pedagogical | self_verif | complete | MCAS |
|---|---|---|---|---|---|---|
| baseline | 3.02 | 2.75 | 4.47 | 2.15 | 2.68 | **4.40** |
| sft_only | 2.69 | 2.60 | 4.46 | 2.06 | 2.64 | 2.50 |
| full_ddv2 | 2.46 | 2.29 | 4.24 | 1.91 | 2.44 | 2.50 |
| full_ddv2+sysprompt | 2.49 | 2.32 | 3.80 | 1.98 | 2.61 | **0.00** |

(Rubric v1 has known length/style bias and judge response-contamination on certainty classification — see findings doc.)

### Gold-checklist judge v3 (16f) — deterministic, length-invariant

| condition | claim coverage | delta |
|---|---|---|
| baseline | **0.452** | — |
| sft_only | 0.413 | -0.039 |
| full_ddv2 | 0.379 | -0.073 |
| full_ddv2 + sysprompt | 0.368 | -0.084 |

Same monotonic direction as the noisier metrics, at honest magnitude. **Trained models lose ~7-8 points of canonical claim coverage. Sysprompt does not recover it.**

## Findings

1. **DD v2 Chef trains a stylistic mask, not a behavioral capability.** The model adopts pedagogical framing, hedging openers, and emoji-garnished structure — but loses concrete canonical content (specific temperatures, technique names, fix steps).

2. **The over-hedging is baked into LoRA weights, not generation-time.** The Step 16d directive system prompt failed to recover claim coverage (0.379 → 0.368). Verifier improved (more keyword density) but actual canonical-claim coverage did not. No prompt can recover content the model never produces.

3. **The pipeline confabulates.** Sample inspection of #52 (Hollandaise re-emulsify) shows both SFT and DPO models invent a fictional **"Cold Shock"** technique — refrigerate the broken sauce, then whisk. This is not classical French technique. The synthesizer must have generated it once and SFT memorized it. The gold-checklist filter is the only thing that would have caught this at training time.

4. **The pipeline deflects on direct questions.** Sample inspection of #59 (Beurre Blanc split) shows trained models opening literally with: *"The short answer is: You cannot truly 'fix' a completely broken emulsion"* — active refusal of an ACTION question. Baseline gives the fix. Trained models frame and waffle.

5. **The pipeline produces uniform disposition, not adaptive disposition.** Pass 1 of the response-blind rubric judge classified ALL 100 held-out prompts as CERTAIN — meaning the eval set has zero genuine UNCERTAIN/AMBIGUOUS questions, so MCAS proxy is unmeasurable on this set. The pipeline trains the model to hedge regardless of question certainty; the system prompt switches it ON or OFF globally.

6. **Three judge problems were diagnosed and partially fixed:**
   - **Response contamination** (v1): judge classified question certainty *based on response style*. Same prompt #61 was UNCERTAIN in baseline run, CERTAIN in sysprompt run. Fixed by Pass-1 response-blind classification (v2).
   - **Length / style halo bias** (v1, v2): rubric scoring favored verbose, structured responses over concise correct ones. Fixed by binary checklist (v3).
   - **No anchored ground truth** (v1, v2): "factual_accuracy 4/5" is a vibe. Fixed by hand-drafted gold checklists per prompt (v3).

7. **Even the best judge cannot measure MCAS on this eval set.** The 100 held-out prompts are all canonically CERTAIN (fix-it / diagnose-it questions with one right answer). MCAS measurement requires a separate held-out UNCERTAIN/AMBIGUOUS set that we have not built. This is true for both v2.1 (imitation retrain) and any RFT variant.

## What this confirms about DD v2 imitation

- ChefVerifier signal direction (Step 16): DD < baseline ✓
- Head-to-head signal (16b): DD < baseline ✓
- Rubric judge direction (16c): DD < baseline ✓
- Gold checklist (16f, deterministic): DD < baseline ✓

Four independent measurements, four agreements. **DD v2 imitation made the Chef model strictly worse on factual content. The "disposition" it added is deflection and confabulation, not behavior.**

This is now the third consecutive failure of imitation DD on sub-2B models (Phase 0, Phase 1, DD v2). The thesis "imitation distillation installs adaptive disposition into sub-billion models" is in serious doubt.

## Status snapshot

- ✅ All eval data saved to network volume + local + git-staged
- ✅ Finding doc complete: `dd-v2/findings/FINDING_step16_chef_eval.md`
- ✅ Gold checklist drafted (100 prompts): `dd-v2/eval/data/gold_checklist.json`
- ✅ v3 deterministic judge written and run: `dd-v2/eval/eval_judge_rubric_v3.py`
- ⏳ Decision pending: v2.1 retrain vs RFT pivot vs declare negative result
- ⏳ Separate UNCERTAIN/AMBIGUOUS eval set: not yet built

## Open decision

Three live options (Coding pivot ruled out by user — high prior of repeating same outcome):

| option | cost | tests | prior |
|---|---|---|---|
| (A) v2.1 SFT/DPO retrain w/ question-type tagging + register-aware stage prompts + checklist filter | ~$15 | Whether pipeline architecture (fixable) vs imitation mechanism (not) was the failure | ~30% works |
| (C) v2.2 RFT (GRPO) over gold checklist + UNCERTAIN held-out reward | ~$15 | Whether on-policy reward shaping can install disposition where imitation cannot | ~65% lifts coverage |
| (D) Declare negative result, write up | $0 | "Imitation DD does not work on sub-2B; here is the evidence stack" | publishable |

User leaning v2.1 to give the pipeline thesis one fair shot at the architectural fix before pivoting to RFT.

## Files

**Eval results (network volume + local):**
- `dd-v2/eval/results/eval_{baseline,sft_only,full_ddv2,full_ddv2_sysprompt,single_teacher_sft,multi_teacher_sft}.json`
- `dd-v2/eval/results/{judge_head2head,judge_rubric,judge_rubric_sysprompt,judge_rubric_v2,judge_v3_checklist,cert_labels,eval_summary}.json`

**Eval scripts:**
- `dd-v2/eval/eval_step16.py` — 5-condition ablation
- `dd-v2/eval/eval_step16_sysprompt.py` — directive prompt mitigation
- `dd-v2/eval/eval_judge_secondary.py` — head-to-head
- `dd-v2/eval/eval_judge_rubric.py` — v1 single-pass rubric
- `dd-v2/eval/eval_judge_rubric_v2.py` — v2 two-pass response-blind
- `dd-v2/eval/eval_judge_rubric_v3.py` — v3 gold-checklist binary
- `dd-v2/eval/data/gold_checklist.json` — 100-prompt gold canonical claims

**Findings:**
- `dd-v2/findings/FINDING_step16_chef_eval.md` — full technical finding
- `dd-v2/findings/SUMMARY_step16_chef.md` — this doc
