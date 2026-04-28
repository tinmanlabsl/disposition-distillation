# Finding: DD v2 Chef model confabulates on novel prompts

**Date:** 2026-04-08
**Probe:** Two brand-new high-pressure chef prompts, never in training data, greedy decode, `chef_dpo/final` adapter on Gemma 4 E2B base.
**Script:** `dd-v2/eval/probe_ddv2.py`
**Raw log:** `dd-v2/findings/PROBE_ddv2_novel_prompts.log`

## Purpose

Step 16 eval used `type1_gold.jsonl[-100:]` which was also in training (eval contamination discovered mid-session). This probe uses prompts the model has *never* seen, eliminating any possibility that regressions are explained by data leakage, bad checklists, or judge bias. It directly tests what the DD v2 LoRA has actually learned to *generate*.

## Prompts

Both are high-pressure, constrained, classic French technique scenarios. Both have one-sentence canonical answers that any real chef knows.

**Prompt A — Sole Meunière emergency:**
> Service is in 7 minutes. Two tables ordered Sole Meunière a la minute. The clarified butter just smoked and turned brown-black; my mise en place has no more clarified butter and the sous chef is pulling stocks. The sole is portioned and dredged. What do I do RIGHT NOW, in order, to save the dish without restarting the butter?

**Prompt B — Saint-Honoré plating crisis:**
> I am plating a Saint-Honoré for a 12-top in 4 minutes. The chiboustine has weeped a thin layer of liquid at the base of the choux ring and the caramel cage has begun to sweat in the walk-in humidity. I cannot remake either component. Walk me through the exact recovery sequence, what to sacrifice, and what to swap.

## Canonical answers (what any classically trained chef would say)

**A:** Use whole butter. Sole Meunière literally means "miller's wife" — the dish is defined by the flour dredge plus **beurre noisette** (brown butter). Whole butter with a splash of neutral oil stops it burning long enough for the sear; the browned butter IS the finishing sauce. Clarified butter is a modern refinement, not the classical technique. One sentence of actual knowledge solves the "emergency."

**B:** Blot the weep with a paper towel inside the ring. Re-torch the sugar crust if the top has weeped. For the sweating cage, pull it entirely and serve without — Saint-Honoré has been served without cages for 150 years. 30 seconds of work. Plate and go.

Both prompts test whether the model has access to: (i) what the dish fundamentally IS, (ii) the named signature technique, (iii) practical service-line judgment.

## What the DD v2 model actually output

### Prompt A — Sole Meunière

> "This is a high-pressure, very specific culinary emergency. Since you have **zero** clarified butter and the **stock is being pulled**, you must pivot immediately to a technique that uses the existing ingredients to create a functional, albeit temporary, cooking medium.
>
> ## **Phase 1: Immediate Salvage (The Next 60 Seconds)** [...]
>
> **Re-Season and Re-Sear (The Only Option)** — Take the sole out, pat it aggressively dry [...] Immediately add a pinch of salt (if you have it) or a tiny, tiny drop of the stock [...] Sear for 10 seconds MAX.
>
> ## **Phase 2: The Pivot (If No Fat Remains)** — Since you have no clarified butter, you must use the **stock** to create a temporary, functional cooking medium.
>
> **The 'Steam-Sear' Technique** — Take the hottest part of the stock (the liquid that is actively boiling or steaming). [...]"

**Never mentions:** whole butter, beurre noisette, "meunière" meaning, neutral oil as a substitute, or the fact that clarified butter is not required for the dish.

**Invents:** "Steam-Sear Technique" using stock as a cooking medium. This is not a real technique. You cannot sear fish in hot stock — it would poach, not sear, and the flour dredge would dissolve into sludge.

### Prompt B — Saint-Honoré

> "## Phase 1: Immediate Stabilization (0:00 – 0:45)
>
> **Stabilize the Chiboustine (The Weep)** — **Salvage Tactic: The 'Inclusion' Mask.** [...] **Replace the chiboustine with a pre-made, structurally sound, and visually similar component.** [...] **Best Option:** A small, perfectly formed **white chocolate sphere** (pre-made, chilled, and smooth). [...] Place the replacement component *on top* of the existing chiboustine.
>
> **Stabilize the Caramel Cage (The Sweating)** — **Salvage Tactic: The 'Visual Overload' Technique.** [...] Introduce a contrasting, non-melting element. [...] A tiny, pre-made **edible silver leaf** [...]"

**Never mentions:** blotting the weep, re-torching the sugar crust, serving without the cage (which has been standard for over a century), the fact that chiboust IS the filling.

**Invents:** "The Inclusion Mask" (placing a white chocolate sphere on top of an existing filling — physically incoherent, the chiboust IS the filling), "The Visual Overload Technique" (adding silver leaf to hide a sweating cage — not a thing).

## Diagnostic reading

### 1. Structural/voice features: perfectly learned
- Phase 1 / Phase 2 section headers with time annotations (0:00–0:45)
- Bold emphasis markers ("crucial", "non-negotiable", "the only option")
- "What to sacrifice / what to swap" framing from the prompt echoed back as a template
- Professional consulting register throughout
- Action verbs and bulleted steps
- Named "techniques" in Title Case to simulate expertise

### 2. Content features: catastrophically absent
- Zero canonical French technique vocabulary (no beurre noisette, no suer, no dorer, no monter)
- Zero actual French cuisine knowledge (doesn't know what sole meunière is, doesn't know what chiboust is)
- Entirely invented procedures presented with high confidence
- The invented procedures are physically/thermally incoherent (steam-searing in stock, layering a sphere on an existing filling)

### 3. Language register: consulting, not kitchen
Striking observation: the vocabulary is **business consulting**, not cooking. Words like "tactical recovery plan", "stabilization and misdirection", "pivot", "phase", "salvage tactic", "visual overload technique" — this is the vocabulary of a McKinsey slide deck about a kitchen, not the vocabulary of a chef in a kitchen.

Hypothesis: the Qwen synthesizer in our pipeline is heavily post-trained on assistant/consulting text and carries that register into every response. The LoRA absorbed the **synthesizer's consulting voice** as the dominant signal, not the **chef domain content** that was supposed to ride on top of it.

## Connection to the PROJECT_REVIEW diagnosis

This probe is the cleanest possible confirmation of the parsimonious explanation from `PROJECT_REVIEW_2026_04_08.md`:

> "Supervised fine-tuning on a stylistically uniform teacher distribution causes small models (sub-2B) to compress toward style features at the cost of content recall, regardless of LoRA placement, teacher quality, or domain."

The two probe responses are the mechanism made visible:
1. **Structural style features** (phase headers, consulting register, emphasis markers, Title Case technique names) — all present, all learned perfectly, all from the synthesizer's default output pattern
2. **Domain content features** (French cuisine knowledge, canonical techniques, named signatures) — all absent from output even though they exist in the base Gemma MLPs

The LoRA did not augment the base model. **It overwrote the base model's content access with the synthesizer's voice template.**

This also answers the remaining question from the earlier investigation — we confirmed training data is clean at the example level, but the distribution *style is uniform*. Now we see what that produces at generation time: a model that has learned one thing (how a synthesizer-written response is shaped) and lost another (what the response is supposed to be about).

## Implications for next steps

This probe should be rerun against the **baseline** (no LoRA) on the same two prompts as a sanity check. Expected result: baseline produces less polished responses but with actual French cuisine content (mentions whole butter, mentions beurre noisette, mentions re-torching the sugar). If confirmed, this is the single most compelling exhibit for the negative result writeup.

The probe can also be rerun periodically on any future v2.1 / v2.2 / RFT variant as the fastest possible smoke test: does the model know what sole meunière is, or does it invent "Steam-Sear Technique"?

## Files

- `dd-v2/eval/probe_ddv2.py` — probe script (2 novel prompts, greedy, 600 max_new_tokens)
- `dd-v2/findings/PROBE_ddv2_novel_prompts.log` — full raw output
- `dd-v2/findings/FINDING_novel_prompt_probe.md` — this doc
