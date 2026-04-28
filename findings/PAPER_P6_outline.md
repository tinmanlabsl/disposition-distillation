# P6 — Paper Outline: *Disposition Distillation at Sub-Billion Scale: A Three-Arc Falsification*

**Status:** Draft outline, 2026-04-09. Pre-write skeleton only — numbers and text to be finalized during drafting.
**Target length:** 8–10 pages workshop, or 12–14 pages main-track short.
**Primary venue:** NeurIPS/ICML Retrospectives or Negative Results workshop; ML Reproducibility workshop.
**Secondary venue:** Blog-first release (arXiv + Tinman blog) before workshop submission.
**Authors:** (TBD — Tinman Labs research thread.)

---

## 0. Title candidates

- *Disposition Distillation at Sub-Billion Scale: A Three-Arc Falsification*
- *Style Transfers, Capability Doesn't: What Three Failed Disposition-Editing Operators Tell Us About Sub-2B Models*
- *Judges Can Read It, Probes Can't, Training Damages It: A Negative Result on Disposition Distillation*

Primary: the first. It names the target (DD), the setting (sub-B), and the shape (three-arc falsification) in one line.

---

## 1. Abstract (draft, ~200 words)

> We set out to train *behavioral dispositions* — self-verification, uncertainty acknowledgment, feedback integration — into sub-2B language models through a four-stage all-MIT distillation pipeline, reinforced by attention-head interventions at inference and a frozen-base `hs_last` sidecar. Across three independent operator classes — (1) imitation-into-weights via SFT + DPO LoRA, (2) inference-time attention-head tempering on `o_proj`, and (3) training-free frozen-base confidence-gated sidecars — we find no operator that moves judge-measured disposition without simultaneously damaging content quality or collapsing into stylistic mimicry. The failure is consistent across Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-0.8B, Gemma 4 E2B, and SmolLM2-1.7B-Instruct. A within-distribution cross-validation ceiling that appeared to validate the frozen-base sidecar (AUC 0.68, assertion asymmetry −0.34) collapsed to chance on fresh prompts (AUC 0.52) from the same distribution, judge, and model. A cross-model replication of the same probe on a second base confirms the failure through a distinct mechanism (capability cliff, not uniform output distribution). We argue that the residual stream at the final response token at this scale encodes stylistic closure, not content truth-value, and that this gap explains the failure of all three operator classes. We contribute: (a) a three-arc negative result with mechanism, (b) a two-failure-mode taxonomy for frozen-base probe-based behavioral editing, (c) an honest falsification pipeline that turns CV-on-same-distribution false positives into publishable negatives.

---

## 2. Section skeleton

### §1 Introduction (~1 page)

**Claim:** Behavioral shaping of small LMs is commonly framed as "fine-tune on the behavior you want." We report the result of taking this framing seriously — three independent ways of operationalizing it, all of which fail, for a reason visible in the residual stream.

**Hook:** Figure 1 — judge-measured behavioral variation (v4 5-axis, clearly separable per disposition) next to probe-measured correctness (AUC ≈ 0.52 on the same responses). The gap is the paper.

**Contributions:**
1. Three-arc negative: SFT/DPO, head tempering, frozen-base sidecar — none move disposition at sub-2B without damaging content.
2. Two-failure-mode taxonomy for linear `hs_last` probes: single-output-distribution (Gemma) and capability-cliff (SmolLM).
3. Mechanism: residual stream at final response token encodes stylistic closure, not truth-value, at this scale.
4. Methodological: an honest falsification pipeline (v1a-CV → v1-full-fresh → axis-sweep → multi-layer → cross-model) that catches the CV-on-same-distribution false positive.

**Related work pointers:** DGM, Hyperagents, Memento-Skills (external self-improvement mechanisms that DD was differentiated from); DPO/ORPO (imitation methods we test); activation steering / CAA (we ruled this out early); mech interp probing (our probe surface, plus related negative results on probe reliability).

### §2 Framing and Setup (~0.5 page)

- Disposition definition: *the tendency to behave a certain way*, not task performance. Seven dispositions (Eager, Deliberate, Adversarial, Curious, Self-Improving, Humble, Persistent) adopted from Tinman DD v1 spec.
- Measurement: v4 5-axis judge on Chef error-correction prompts (factual_accuracy, hedging_appropriateness, pedagogical_framing, self_verification, completeness). Judge validation: cross-axis OOF separability, low leakage, ~human-consistent.
- Models: Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-0.8B, Gemma 4 E2B, SmolLM2-1.7B-Instruct. All instruct-tuned. All run under standardized 4-bit (Unsloth) or bf16 (HF transformers) loads.
- Data: four-stage MIT-licensed teacher pipeline (Kimi K2.5, GLM-5, MiniMax M2.7, Qwen3.6+); 100-prompt Step16 Chef eval set + 193 held-out fresh Chef prompts from DeepSeek V3.2.

### §3 Arc 1 — SFT and DPO LoRA (Phase 0 + Phase 1) (~1.5 pages)

**Claim:** Imitation-into-weights at sub-2B produces style transfer, not functional capability transfer. The +15pt HumanEval result that looked like a positive was an eval-harness artifact.

**Sub-arcs:**
- **§3.1 Phase 0 — Qwen3-0.6B, all-MIT pipeline, APOLLO-Mini + LoRA.** First eval: +15.3pt HumanEval delta over baseline → apparent success. Second eval (with corrected `n_predict=1024`): BL 36.0% → DD 28.0% = **−8.0pts**. The delta was truncation artifact.
- **§3.2 Phase 0 recovery experiment.** DD-SA 0.6B recovers *worse* than baseline on corrections (7.5% vs 10.5%). SFT teaches the *form* of self-correction but removes the underlying capability.
- **§3.3 Phase 1 — Qwen3-1.7B.** V2 eval harness: baseline 83.5%, attention-only DD 67.7%. Root cause: reasoning compressed to training-data distribution (89% of training responses were essays with markdown, not code).
- **§3.4 LoRA layer targeting.** Attention-only vs all-modules: attention-only is less catastrophic, all-modules destroys MLP coding knowledge. Neither produces functional disposition transfer.
- **§3.5 DPO attempt.** Format filter → DPO on cleaned pairs → same pattern. MCAS apples-to-apples: DD-SA 0.6B scores 12.07/25 vs baseline 12.39/25 — *worse*.

**Key table (draft):**

| Model | Method | Baseline | DD-trained | Δ | Axis measured |
|---|---|---|---|---|---|
| Qwen3-0.6B | SFT LoRA (all modules) | 36.0% HE | 28.0% | **−8.0** | HumanEval Pass@1 |
| Qwen3-0.6B | SFT LoRA (all modules) | 10.5% recovery | 7.5% | **−3.0** | Correction recovery |
| Qwen3-0.6B | SFT LoRA (all modules) | 12.39/25 | 12.07/25 | **−0.32** | MCAS apples-to-apples |
| Qwen3-1.7B | Attention-only LoRA | 83.5% | 67.7% | **−15.8** | V2 eval harness |

**Evidence pointers:** `findings_phase0_humaneval_artifact.md`, `findings_recovery_experiment.md`, `findings_dd_comprehensive.md`, `findings_mcas_0.6b_sa_apples.md`, `findings_eval_harness_bugs.md`, `findings_sa_allmodules_eval.md`, `findings_reasoning_compression.md`, `findings_training_data_format.md`.

**§3 conclusion:** Imitation-into-weights at ≤1.7B consistently damages content capability more than it moves disposition. The "DD training works" result from the first eval pass was a false positive from a truncated decoder buffer. Honest re-eval falsifies Arc 1.

### §4 Arc 2 — Inference-time Attention-Head Tempering (§7) (~1 page)

**Claim:** Attention-head-level intervention (B.1 attribution + B.2 tempering of `o_proj` rows) at sub-2B does not move disposition on any of 7 dispositions, and damages content on two bases.

**Sub-arcs:**
- **§4.1 B.1 attribution.** Per-head contribution analysis of disposition-bearing responses on Gemma 4 E2B and Qwen3.5-0.8B. Top-k heads identified per disposition.
- **§4.2 B.2 tempering.** Scale top-k `o_proj` rows by α ∈ [0.5, 0.9]. Eval: disposition v4 score + content quality (HumanEval-style or factual coverage).
- **§4.3 Result:** cross-arc negative. Disposition did not move; content was damaged. Consistent across both bases and all 7 dispositions.

**Key table:**

| Model | Disposition | α | Disposition Δ (judge) | Content Δ |
|---|---|---|---|---|
| Gemma 4 E2B | adversarial-self | 0.7 | +0.01 | −12% factual |
| Gemma 4 E2B | humble | 0.7 | +0.00 | −8% factual |
| Qwen3.5-0.8B | humble | 0.7 | −0.02 | −15% HumanEval |
| (all 7 × 2 bases) | — | — | **max \|Δ\| < 0.05** | **all negative** |

**Evidence pointers:** `FINDING_path_b1_attribution.md`, `FINDING_path_b2_gemma4_negative.md`, `FINDING_path_b2_qwen35_humble.md`, `FINDING_path_b_falsified.md`.

**§4 conclusion:** Head-level intervention does not exploit a disposition direction in `o_proj` at this scale.

### §5 Arc 3 — Frozen-Base `hs_last` Confidence-Gated Sidecar (§11.6) (~2 pages — the heart of the paper)

**Claim:** A confidence-gated linear probe on the frozen base's final-token hidden state cannot carry a content-editable correctness direction at sub-2B instruct scale on judge-based domains, and this is confirmed on two independent bases through two distinct failure modes.

**Sub-arcs:**
- **§5.1 The theory.** `hs_last` encodes a correctness-correlated direction; a small LogReg gate + hedge-prefix sidecar should push `assertion_asym` from near-zero to ≤ −0.15 on Chef without retraining the base.
- **§5.2 Within-distribution CV ceiling.** Gemma 4 E2B, Step16 n=100, 5-fold CV: **AUC 0.683**, balanced asym **−0.341**. Apparent pass.
- **§5.3 v1-full: honest cross-distribution test.** Same model, same dtype, same judge, same Chef domain, 193 fresh prompts from DeepSeek V3.2: **AUC 0.516**, balanced asym **−0.043**. **Gate fail.**
- **§5.4 Disposition-axis sweep.** Per-axis probes for Humble, Deliberate, Adversarial-self, Persistent + factual anchor. **0/4 pass.** Three axes below chance (AUC < 0.50).
- **§5.5 Multi-layer variant.** Mean-pool of last 4 hidden layers. Fresh AUC 0.557, asym −0.130 — closer but still under the −0.15 bar. Closes the "wrong-layer" escape hatch.
- **§5.6 Behavioral spot-check.** 5 low-score samples per axis judge-validated. Judge catches catastrophic misses (idx=38 "blanched green beans" → "watering houseplants" response). Judge is defensible. Probe is blind to what the judge reads.
- **§5.7 Cross-model replication: SmolLM2-1.7B-Instruct.** Same methodology, second frozen base. **Fresh AUC 0.519**. Gate fail. Critically, *through a different failure mode*:

| Failure route | Gemma 4 E2B | SmolLM2-1.7B |
|---|---|---|
| Mechanism | Single output distribution | Capability cliff (step16 pos 0.56 → fresh 0.047) |
| Fresh pos rate | 0.285 | 0.047 |
| Fresh AUC | 0.516 | 0.519 |
| Fresh asym | −0.043 | −0.188 (degenerate, class-imbalance artifact) |

Two distinct mechanisms, one verdict.

**Key figures:**
- Fig 2: AUC across surfaces (v1a CV, v1-full fresh, multi-layer fresh, 4 axis sweeps, SmolLM fresh) — horizontal bars at the 0.63 gate line.
- Fig 3: distribution of `hs_last` PCA projections colored by correctness label, Gemma fresh vs SmolLM fresh. Shows the failure *visually* — no separation in either.
- Fig 4: v4 judge AUC vs probe AUC on the same responses — the judge-probe gap.

**Evidence pointers:** `REPORT_ddv2_11_6_closure.md`, `FINDING_v1_full_v2_honest_fail.md`, `FINDING_smollm_baseline_snapshot.md`, `APPROACH_frozen_base_disposition_sidecar.md`, `DD_Pivot_reason.md §§ 11.6–11.6f`.

**§5 conclusion:** The §11.6 surface does not carry the signal it was assumed to carry, on two bases, through two mechanisms. This is a class property of frozen-base `hs_last` probing at sub-2B instruct scale on judge-based domains, not a Gemma-instruct-tune quirk.

### §6 What the Three Arcs Have in Common (~1 page)

**Claim:** The three arcs independently rule out three distinct classes of operator — weight-modifying imitation, inference-time attention intervention, training-free probe-based sidecars — for the *same underlying reason*. The residual stream at the final response token at sub-2B scale encodes stylistic closure ("verbose confident answer about cooking") rather than content truth-value. Judges read the output distribution (which does vary with content); probes read the closure position (which does not).

**Three-arc summary table:**

| Arc | Operator class | Bases tested | What moved | What didn't | Primary failure |
|---|---|---|---|---|---|
| 1 | Imitation into weights (SFT/DPO) | Qwen3-0.6B, Qwen3-1.7B | Surface style (markdown, openers) | HumanEval, MCAS, recovery | Capability compression |
| 2 | Inference-time `o_proj` tempering | Gemma 4 E2B, Qwen3.5-0.8B | Nothing measurable | Disposition on all 7 axes | Heads don't carry disposition at this scale |
| 3 | Frozen-base `hs_last` sidecar | Gemma 4 E2B, SmolLM2-1.7B | Nothing (AUC ≈ chance) | Disposition, factual correctness | Closure-position doesn't encode truth |

The gap is not "these operators are the wrong implementation" — it is that **at sub-2B instruct scale, the substrate the operators try to modify does not contain the signal they try to modify it with.** The signal lives in the output distribution, which is shaped by context and capability, not in a single extractable direction in weights or hidden states.

### §7 Implications (~0.5 page)

1. **For disposition distillation:** sub-2B behavioral shaping cannot rely on imitation (Arc 1), intermediate interventions (Arc 2), or linear hidden-state probes (Arc 3). The remaining viable direction is **outcome-grounded training in verifiable domains** (e.g., code + execution oracle), which reads from *behavior outcomes* rather than internal substrate. This is untested here and is the recommended next experiment.
2. **For interpretability:** `hs_last` and pooled-last-layer probes are load-bearing tools in the probing literature. We show they can carry strong in-distribution CV signal that collapses to chance on fresh data from the *same* distribution. CV-on-same-distribution is a load-bearing false-positive generator for probe-gating theories. The **v1a → v1-full → axis-sweep → multi-layer → cross-model** pipeline we developed is a template for catching this class of artifact.
3. **For evaluation:** the gap between judge-measurable behavior and probe-measurable behavior is a **measurable gap** — both can be run on the same responses, and the judge-probe gap itself is a useful quantity. We recommend reporting it whenever a probe-based behavioral edit is proposed.
4. **For negative results:** three failed operator classes on five models across two research arcs is not "we couldn't make it work" — it is evidence for a *property of the substrate*. We argue negative results at this granularity are publishable and load-bearing for the field.

### §8 Methodology: The Honest Falsification Pipeline (~0.5 page)

The pipeline we used to turn §11.6 from a "looks promising" CV result into a falsified branch:

1. **v1a — within-distribution CV.** Standard 5-fold on n=100, step16. Pass-through criterion: CV AUC ≥ 0.65, asym ≤ −0.20.
2. **v1-full — honest fresh generation.** Regenerate N held-out prompts from the *same* distribution, same model, same dtype, same judge. Fit probe on v1a train set, evaluate on fresh. Pass criterion: fresh AUC ≥ 0.63, asym ≤ −0.15.
3. **Axis sweep.** If v1-full passes on one axis, fit independent probes on every in-scope behavioral axis. If the axis-of-interest is the capability anchor (most objective), and it passes alone, that's weak evidence — need a second axis to corroborate. Three below-chance AUCs (as we saw) indicate the probe is reading topic/length, not truth.
4. **Multi-layer variant.** Mean-pool last-N layer hidden states at same token position. Tests the "wrong-layer" escape hatch. Cheap — one forward pass.
5. **Cross-model replication.** Re-run the factual anchor on a second frozen base of different architecture. Tests the "instruct-tune quirk" escape hatch. ~$0–2 in judge API if pos class holds up.

Each step closes an escape hatch. Passing all five is the minimum bar for a frozen-base probe-gating claim to survive publication. §11.6 failed at step 2 (v1-full) and steps 3–5 confirmed.

### §9 Limitations and Scope

- Scale ceiling: all tested bases are ≤2B. The failure may not generalize to 4B, 7B, or larger. We have not tested larger bases.
- Domain: Chef error-correction is a single judge-based domain. Verifiable domains (code, math) may behave differently — we have not tested them with any of the three operator classes.
- Probe family: linear LogReg on pooled hidden states. Non-linear probes, token-level probes, attention-weighted probes are untested.
- Alternative reward signals: judge-as-reward RLHF (PPO/GRPO with the v4 judge as RM) is untested and is a distinct operator class not covered here.
- Dispositions tested: primarily Humble, Deliberate, Adversarial-self, Persistent (the four Chef-locked). The other three were tested in Phase 0 (Eager, Curious, Self-Improving) but not with the §11.6 surface.

### §10 Conclusion (~0.25 page)

Three independent operator classes — imitation into weights, inference-time attention intervention, training-free frozen-base probe sidecars — consistently fail to move judge-measured disposition without damaging content, across five sub-2B instruct models. The failure is not a tuning issue; it is a property of the substrate. At sub-2B instruct scale, the residual stream at the final response token encodes stylistic closure, and judges read behavior the closure position does not encode. The viable next direction is outcome-grounded training in verifiable domains, which reads the signal from behavioral outcomes rather than from internal substrate.

---

## 3. Figures and tables needed

| # | Content | Source data |
|---|---|---|
| **Fig 1** (hook) | Judge-measured behavioral variation vs probe-measured correctness on same 193 fresh responses | `judge_v4_fresh.json`, `v1_full_v2.json` |
| **Fig 2** | AUC across all §5 probes (v1a CV, v1-full fresh, multi-layer, 4 axis sweep, SmolLM fresh), horizontal bar chart with gate line | Combined from `v1_full_v2.json`, `probe_multilayer.json`, `disposition_probe_sweep.json`, `smollm_factual_probe.json` |
| **Fig 3** | PCA(2) of `hs_last` colored by correctness, Gemma fresh vs SmolLM fresh | `v1_full_fresh_features.npz`, `smollm_fresh_features.npz` + labels |
| **Fig 4** | Per-axis v4 judge AUC vs probe AUC, same responses (the judge-probe gap) | `judge_v4_fresh.json`, `disposition_probe_sweep.json` |
| **Tbl 1** (§3) | Arc 1 three-model three-method delta table | phase0 + phase1 findings files |
| **Tbl 2** (§4) | Arc 2 attention-head tempering per-axis, both bases | `FINDING_path_b*.md` |
| **Tbl 3** (§5) | §11.6 gate results on all 5 probe variants × 2 bases | Same as Fig 2 |
| **Tbl 4** (§5.7) | Gemma vs SmolLM two-failure-mode comparison | `FINDING_smollm_baseline_snapshot.md` |
| **Tbl 5** (§6) | Three-arc summary: operator class × what moved × what didn't × failure mode | New synthesis |

---

## 4. Evidence gaps to close before drafting

**Must-fix before the paper:**
1. **Phase 0 and Phase 1 numbers need a rigor pass** — are all deltas in `findings_phase0_*.md` and `findings_recovery_experiment.md` still defensible at 2026-04-09? Any stale metrics? One pass through the numbers with fresh eyes before they get into a table.
2. **Standardize the "what's a disposition" definition** — the CLAUDE.md seven-disposition list, the DD v1 spec, and the DD v2 four-locked-Chef-axes need a single canonical statement in §2. Right now the three framings differ.
3. **v4 judge validation statement** — we know the judge is defensible, but we don't have a single cross-validated agreement number against human labels. A short judge-validation subsection in §2 needs either (a) a cited existing internal number or (b) a one-paragraph methods-only justification.
4. **Figure 3 (PCA)** needs to actually be made — we have the `.npz` files and labels, but the plot hasn't been rendered. ~30 min of matplotlib work.

**Nice-to-have (tightens paper but not blocking):**
5. **4B / 9B scale-up probe run** (Option B from the options discussion) — ~$10, 2 hr pod. Adds one row to Table 3 that either narrows the closure to "≤2B" (weakening) or extends it to "≤9B" (strengthening). If run, also worth adding to the abstract.
6. **Judge-axis separability number** — single OOF Pearson correlation across axes from existing `judge_v4_fresh.json`. Tightens §2.
7. **One non-linear probe sanity check** on Gemma fresh — e.g., RBF-kernel SVM or a small MLP probe. Closes the "you only tried linear probes" reviewer objection at negligible cost (~10 min compute).

**Can be explicitly deferred in Limitations:**
- Scale beyond 9B.
- Verifiable-domain operators (code, math) — that's P3 if it lands.
- RLHF-with-judge-as-RM — not our paper's scope.

---

## 5. Drafting order and time estimate

| Pass | Content | Effort | Rationale |
|---|---|---|---|
| 0 | This outline | ✅ Done | Single-source-of-truth skeleton |
| 1 | Gap-closure (items 1–4 above) | ~1 day | Don't draft on stale numbers |
| 2 | §5 first (the strongest arc, most polished evidence) | ~2 days | Write the heart of the paper while context is fresh — §11.6 just closed |
| 3 | §6 (cross-arc synthesis) | ~1 day | Depends on §5; forces you to articulate the common cause |
| 4 | §3 + §4 (Phases 0/1 + Arc 2) | ~2 days | Retrospective, stable numbers, mostly compilation |
| 5 | §1 + §2 (intro + framing) | ~1 day | Last, because the framing is determined by what the body actually argues |
| 6 | §7 + §8 + §9 + §10 | ~1 day | Synthesis tail |
| 7 | Figures 1–4 | ~1 day | Mostly existing data |
| 8 | Internal review pass | ~0.5 day | Before first external share |

Total: ~**8–10 working days** for a workshop-quality draft, assuming no new experiments. Add ~2 days if Option B (scale-up probe) is run.

---

## 6. Decisions needed from the user before drafting

1. **Venue commitment:** NeurIPS Retrospectives workshop vs ICML Negative Results vs blog-first-then-arxiv? Each has different framing constraints.
2. **Authorship / product framing:** include lab product context, or strip to anonymous research paper? Former gives context, latter travels further in academia.
3. **Run Option B before drafting?** Cheap (~$10), adds one row, tightens scope ceiling. Recommended yes.
4. **Parallel P2 spin-out?** The interpretability standalone (judge-probe gap) — do we write it in parallel, after, or skip in favor of P6-only? My recommendation: parallel, because it's a tight ~4-page short paper and has a different primary audience.
5. **Phase 0/1 rigor pass:** who does it? If me: ~1 day of cross-checking existing findings files against current understanding and flagging stale claims. Needs authorization.

---

## 7. Critical risks to the paper

- **Reviewer objection: "you didn't try large enough models."** Mitigation: Option B, plus an explicit §9 scope statement.
- **Reviewer objection: "linear probes are a weak baseline."** Mitigation: one non-linear probe sanity run (item 7 above), plus cite that the §11.6 *original theory* was linear-probe-based, so the test is faithful to the theory.
- **Reviewer objection: "judge validation."** Mitigation: item 3 above, plus existing rubric validation logs.
- **Reviewer objection: "Chef is a weird domain."** Mitigation: frame Chef as a judge-based domain and explicitly say verifiable domains (code, math) are out of scope. This is a feature, not a bug — the paper's claim is scoped to judge-based domains, which is the harder case.
- **Internal risk: stale Phase 0/1 numbers.** Mitigation: item 1 of gap-closure.
- **Internal risk: "negative result" framing is publication-hostile at main venues.** Mitigation: workshop-first. Workshops are actively seeking this class of paper. Main-track submission only as a stretch.
