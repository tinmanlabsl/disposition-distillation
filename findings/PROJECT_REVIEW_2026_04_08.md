# DD Project — Step-Back Review

**Date:** 2026-04-08
**Author:** Session-end honest review after Step 16 + LoRA architecture investigation
**Trigger:** User asked: "step back, look at what we have been trying to do, look at what we found, and our current assumptions and approach. See if it really makes sense."

---

## What we set out to do

Per `CLAUDE.md`, the DD project goal:

> "Disposition Distillation distills *how a model behaves* (self-verification, uncertainty acknowledgment, feedback integration) into weights, not system prompts. Related work builds external self-improvement mechanisms; DD trains the *tendency* to improve into weights."

Concrete plan: 7 sub-billion specialist models trained via 4-stage MIT teacher pipeline (Eager → Deliberate → Adversarial → Synthesizer), evaluated on MCAS/SVR/IIS/FIS, target CDS ≥55%, publishable at NeurIPS workshop or arXiv. Total budget ~$545-735 over 6-7 weeks. Phase 0 was the validation gate.

## What actually happened across three independent runs

| run | model | size | LoRA configs tried | reported | corrected/honest |
|---|---|---|---|---|---|
| Phase 0 (coding) | Qwen3-0.6B | 0.6B | attn-only, all-modules | "+15.3 pts HumanEval" | **-8 pts** (n_predict=512 truncation artifact) |
| Phase 1 (student-aware coding) | Qwen3-1.7B | 1.7B | attn-only, all-modules, r=16/32, think-mask, DPO | "Phase 1 complete" | best variant **-15.8 pts** (67.7 vs 83.5 baseline) |
| DD v2 (French chef) | Gemma 4 E2B | 2.3B effective | attn-only r=64 + DPO | Step 16 in progress | **-7.3 pts gold checklist** (0.379 vs 0.452) |

### Cross-run pattern

- **3 model families** (Qwen3 0.6B, Qwen3 1.7B, Gemma 4 E2B)
- **3 domains** (Python coding, student-aware coding, French cuisine)
- **3 training paradigms** (raw SFT, student-aware SFT, student-aware SFT+DPO)
- **5+ LoRA configurations** across runs
- **Identical outcome:** trained model regresses on actual task content vs base model

Three independent falsifications of the imitation-DD claim is no longer "three problems to fix individually." It is a pattern.

## What we learned in DD v2 specifically

The DD v2 chef investigation produced two findings that the prior phases did not have evidence for:

### Finding 1: Training data is clean at the example level

Inspected synthesizer outputs for the 5 worst-failing eval prompts (#5 Mirepoix, #23/#52 Hollandaise, #29 Soufflé, #59 Beurre Blanc). All five training targets contain the canonical fix:
- #59 Beurre Blanc: *"remove the pan from the heat entirely. This is non-negotiable."* — concrete fix with the exact 88°C danger threshold
- #52 Hollandaise: *"two reliable ways to bring it back"* — promises both methods, then provides them
- #5 Mirepoix: names *suer*, gives 250-300°F sweat range
- #29 Soufflé: names two specific structural causes with mechanism

**The data is not broken.** "Pipeline confabulation" was my hypothesis earlier in the session and it was wrong. The synthesizer wrote correct canonical answers.

### Finding 2: Eval contamination is present and irrelevant

The eval used `type1_gold.jsonl[-100:]` as a "held-out tail." That tail is **inside the training set**. All 100 eval prompts (lines 584-683) were in DD v2 training. The model saw the correct answer at training time AND still produced deflection at inference. This is not a generalization gap. It is **active information loss during training**.

### Finding 3: The deflection is in LoRA weights, not generation

Step 16d sysprompt mitigation experiment: re-ran full_ddv2 with an explicit anti-deflection directive prompt. Verifier improved (more keyword density), gold checklist did NOT improve (0.379 → 0.368). No prompt recovers content the model has been trained not to produce.

### Finding 4: The base model already knows everything DD was supposed to teach

- Baseline gold checklist coverage: 0.452 — without any DD training, Gemma 4 E2B knows the canonical fixes
- Baseline produces direct concrete answers under no system prompt
- Sysprompt experiment showed the base model can be made humble or direct **by prompting alone**

The disposition DD is trying to install is **already present** in the base model and is **already promptable**. DD is not installing new capability; it is attempting to bake in a default.

### Finding 5: LoRA configuration was attn-only, MLP frozen

```python
finetune_attention_modules=True,
finetune_mlp_modules=False,
r=64, lora_alpha=128
```

The base Gemma 4 E2B's MLPs already contain the canonical knowledge. The LoRA, frozen out of MLP, can only modify *attention routing*. It cannot install new content; it can only re-route attention over existing MLP features. When training data contains both style and content signals, attention-only LoRA is forced to choose what to encode, and stylistic features (rank-cheap, uniform across training set) win over content features (rank-expensive, non-uniform).

**However:** Phase 0 and Phase 1 both tested all-modules LoRA. Both failed. So "switch LoRA structure" as a class of fix has been falsified at least 4 times across the project — only the specific variant "frozen MLP + parallel adapter + attn LoRA" remains untested.

## Most parsimonious explanation

The single hypothesis that fits all the evidence across 3 phases:

> **Supervised fine-tuning on a stylistically uniform teacher distribution causes small models (sub-2B) to compress toward style features at the cost of content recall, regardless of LoRA placement, teacher quality, or domain.**

Mechanism:
1. Training data has uniform stylistic surface (every response: warm opener, framing, headers, emojis, "wipes hands on apron")
2. Training data has non-uniform content (each fix is different)
3. SFT optimizes the model toward the average training distribution
4. Style averages high (uniform) → model emits it consistently
5. Content averages low (varied) → model loses access to it
6. The base model's MLP content knowledge becomes inaccessible because attention has been re-routed toward style features
7. Result: structurally faithful output, factually empty

This is not a bug in any single component. It is how SFT on stylistically uniform data works on capacity-constrained models. The pipeline architecture, the teacher choice, the LoRA placement — none of these are the root cause. The root cause is the *combination* of (a) imitation as the training mechanism, (b) stylistic uniformity in the training distribution, and (c) sub-2B model capacity.

## Implications for the broader project

If the parsimonious explanation is correct:

1. **Phase 0 was not a successful validation.** The +15.3 pts result was a measurement artifact. The corrected result is -8 pts. Phase 0 is a failed validation that was reported as success and the project proceeded on a false positive.

2. **The Phase 1-6 plan is conditional on a working DD foundation.** BitNet quantization (Phase 3), RYS amplification (Phase 0.5), the 7-model release (Phase 2 + 6), all assume DD installs disposition. Three failures suggest those phases are building on sand.

3. **The disposition is already in the base model and is already promptable.** The sysprompt experiment showed Gemma 4 E2B can be humble or direct depending on prompting. DD is not adding capability; it is removing flexibility (forcing one mode instead of context-aware switching). This may be an *anti-pattern*, not a feature.

4. **The remaining unexplored variants are increasingly narrow.** v2.1 pipeline fixes (data is fine), v2.1b parallel MLP adapter (one untested LoRA structural variant out of many already failed), style-randomized SFT (untested but cheap). None of them attack the parsimonious root cause.

5. **The single unexplored mechanism that *would* attack the root cause is RFT/GRPO** — on-policy gradient flows from the model's own outputs against a verifier reward, breaking the SFT-compression-toward-uniform-style dynamic. RFT is well-trodden (DeepSeek-R1, Tulu-3) so it does not preserve the DD novelty claim, but it is the only remaining mechanism that could install behavior into weights.

## Confidence updates

| claim | prior (start of session) | posterior (now) |
|---|---|---|
| Imitation DD installs adaptive disposition into sub-2B models | ~50% | **~10%** |
| The DD v2 pipeline can be fixed with architectural pipeline tweaks (tagging, register prompts, deflection check) | ~40% (when I drafted PLAN_v2.1) | **~5%** (data was clean; pipeline isn't the problem) |
| LoRA structural change (parallel MLP adapter) will fix v2.1 | implied ~50% (when I proposed v2.1b) | **~20-25%** |
| Style-randomized SFT will fix the regression | not considered | **~40%** |
| RFT/GRPO with verifier reward will lift content coverage | ~65% | **~55-60%** (still high but the same compression dynamic could partially apply) |
| The base model's promptable disposition is already sufficient and DD is anti-flexibility | ~10% | **~50%** |
| Negative result across 3 phases is itself a publishable contribution | ~30% | **~70%** |

## What I was wrong about in this session

In the interest of honesty about my own track record:

1. **First half of session:** confidently claimed the failure was rubric judge bias. Built v2 two-pass response-blind judge to fix it. The judge fix was real and useful, but the regression remained at honest magnitude (-7.3 pts). I anchored on "the model is fine, the judge is bad" longer than the data warranted.

2. **Mid session:** confidently claimed the synthesizer hallucinated "Cold Shock" and the data was the root cause. Drafted PLAN_v2.1 around pipeline fixes. **The data is clean. I had not checked.**

3. **Late session:** confidently claimed the failure was attn-only LoRA and proposed v2.1b parallel MLP adapter. The hypothesis is consistent with the local DD v2 evidence but **inconsistent with Phase 0/1 priors** where multiple LoRA configurations were already tested and all failed. I had not cross-checked against the project history.

Pattern: each "fix proposal" was locally consistent with the data immediately in front of me and inconsistent with the broader project history. Step-back review caught this. Going forward I should cross-check new hypotheses against ALL prior phase results, not just the current run.

## Honest options going forward

| option | cost | tests | my honest prior of working |
|---|---|---|---|
| (1) v2.1b parallel MLP adapter (close out LoRA structural branch) | ~$5 | Last LoRA variant we haven't tried | 20-25% |
| (2) Style-randomized SFT (5+ teacher voices, no uniform "warm chef") | ~$10 | Whether stylistic uniformity is the specific compression mechanism | 40% |
| (3) v2.2 GRPO/RFT over gold checklist | ~$15 | Whether on-policy training breaks the SFT-compression dynamic | 55-60% lift, ~30% adaptive disposition |
| (4) Reframe DD as runtime disposition routing (no training, prompt + light routing) | $0 + reframe | Whether the base model is already sufficient | high — sysprompt evidence supports this |
| (5) Declare imitation DD falsified, write up negative result | $0 | Nothing new; the falsification is already in the data | publishable as-is |

## Recommendation

**Path I would take:** combine (4) and (5). Treat the three-phase falsification as definitive. Reframe the project around the empirical finding that **the base sub-2B model already has the disposition and only needs runtime routing to elicit it**. This is consistent with all the evidence — including the positive Phase 0/1/v2 result that we kept missing: *the baseline always wins*.

**The actual research contribution becomes:** "Sub-billion models do not need disposition distillation. They need disposition *routing*. Here is the empirical evidence across 3 domains and 3 model families that imitation distillation reduces flexibility rather than increasing it, and here is a runtime routing approach that achieves the disposition behaviors via prompt + light orchestration."

**Cost:** ~$0 to test the claim (already have all the data we need)
**Novel contribution:** large — pivots from "training is the answer" to "elicitation is the answer" with empirical justification
**Reuses:** all existing eval infrastructure, gold checklists, judges
**Falsifies:** the original DD thesis, but in a way that produces a stronger downstream story

**Path I would NOT take:** another LoRA variant. The branch is exhausted.

**Path I am genuinely uncertain about:** RFT (option 3). It is the only remaining training-based approach that could plausibly work, and 55-60% prior is not low. But it does not preserve the DD-as-novel-thesis story — the contribution becomes "we used GRPO on a custom verifier, like everyone else." The question is whether the goal is "validate the original DD thesis" (which is now very unlikely to succeed) or "produce a useful research contribution from this work" (which the reframe in option 4+5 does without needing another training run).

## What needs user decision

1. **Accept the three-phase falsification of imitation DD?** (yes/no — drives everything else)
2. **If yes:** pursue reframe (4+5), pursue RFT (3), or both?
3. **If no:** which single closeout experiment justifies the spend (1, 2, or 3)?
4. **What to do with the existing project plan in CLAUDE.md?** It is currently planning Phase 1-6 around an assumption that has been falsified 3 times.

## Files committed

- `dd-v2/findings/FINDING_step16_chef_eval.md` — full Step 16 technical
- `dd-v2/findings/SUMMARY_step16_chef.md` — Step 16 executive summary
- `dd-v2/findings/PROJECT_REVIEW_2026_04_08.md` — this doc
- `dd-v2/eval/results/*.json` (13 files) — all eval data
- `dd-v2/eval/data/gold_checklist.json` — 100-prompt gold canonical claims
- `dd-v2/eval/eval_judge_rubric_v3.py` — deterministic checklist judge
