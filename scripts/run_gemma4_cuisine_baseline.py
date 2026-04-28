#!/usr/bin/env python3
"""
Gemma 4 E2B — French Cuisine Baseline Knowledge Test

Tests whether Gemma 4 E2B already has strong French cuisine knowledge.
If it scores very high (>0.8), there's no room for DD to show improvement,
making it a bad test case for disposition distillation.

Runs 50 sampled prompts across all categories/difficulties,
scores against the reference DB, reports aggregate stats.
"""
import json, time, random, signal, os, sys
signal.signal(signal.SIGHUP, signal.SIG_IGN)

# ---- paths ----
MODEL_PATH = "/workspace/persistent/models/gemma-4-e2b"
PROMPTS_PATH = "/workspace/persistent/cuisine_test/chef_prompts.jsonl"
REF_DB_PATH = "/workspace/persistent/cuisine_test/french_cuisine_reference.json"
VERIFIER_PATH = "/workspace/persistent/cuisine_test/chef_verifier.py"
RESULTS_PATH = "/workspace/persistent/eval/gemma4_cuisine_baseline.json"

N_SAMPLES = 50
random.seed(42)


def load_prompts():
    prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def stratified_sample(prompts, n):
    """Sample n prompts, balanced across categories."""
    by_cat = {}
    for p in prompts:
        cat = p.get("category", "unknown")
        by_cat.setdefault(cat, []).append(p)

    per_cat = max(1, n // len(by_cat))
    sampled = []
    for cat, items in sorted(by_cat.items()):
        random.shuffle(items)
        sampled.extend(items[:per_cat])

    # Fill remaining slots randomly from leftovers
    used_ids = {p["id"] for p in sampled}
    remaining = [p for p in prompts if p["id"] not in used_ids]
    random.shuffle(remaining)
    while len(sampled) < n and remaining:
        sampled.append(remaining.pop())

    return sampled[:n]


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Add verifier directory to path
    sys.path.insert(0, os.path.dirname(VERIFIER_PATH))
    os.environ["CHEF_REF_DB"] = REF_DB_PATH
    from chef_verifier import ChefVerifier

    verifier = ChefVerifier(db_path=REF_DB_PATH)

    # Load prompts and sample
    all_prompts = load_prompts()
    print(f"Loaded {len(all_prompts)} prompts. Sampling {N_SAMPLES} (stratified).", flush=True)
    sampled = stratified_sample(all_prompts, N_SAMPLES)

    cat_counts = {}
    for p in sampled:
        cat = p.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"Sample distribution: {cat_counts}", flush=True)

    # Load model
    print(f"\nLoading Gemma 4 E2B from {MODEL_PATH}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    # Run inference + verification
    results = []
    scores_by_cat = {}
    scores_by_diff = {}
    total_score = 0.0
    start = time.time()

    for i, p in enumerate(sampled):
        prompt_text = p["prompt"]
        cat = p.get("category", "unknown")
        diff = p.get("difficulty", "unknown")

        messages = [{"role": "user", "content": prompt_text}]
        input_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tok([input_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

        # Verify
        score = verifier.score_response(response, prompt_text)
        verification = verifier.verify_response(response, prompt_text)

        total_score += score
        scores_by_cat.setdefault(cat, []).append(score)
        scores_by_diff.setdefault(diff, []).append(score)

        results.append({
            "id": p["id"],
            "category": cat,
            "difficulty": diff,
            "prompt": prompt_text[:200],
            "response": response[:500],
            "score": score,
            "passed": verification["passed"],
            "issues": verification["issues"],
            "response_length": len(response),
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            avg = total_score / (i + 1)
            rate = (i + 1) / elapsed * 60
            print(f"  [{i+1}/{N_SAMPLES}] avg_score={avg:.3f} | {rate:.0f} prompts/min", flush=True)

    elapsed = time.time() - start
    avg_score = total_score / len(results)

    # Aggregate stats
    cat_avgs = {c: sum(s)/len(s) for c, s in scores_by_cat.items()}
    diff_avgs = {d: sum(s)/len(s) for d, s in scores_by_diff.items()}
    n_passed = sum(1 for r in results if r["passed"])

    summary = {
        "model": "gemma-4-e2b",
        "n_prompts": len(results),
        "avg_score": round(avg_score, 4),
        "verification_pass_rate": round(n_passed / len(results), 4),
        "scores_by_category": {c: round(v, 4) for c, v in sorted(cat_avgs.items())},
        "scores_by_difficulty": {d: round(v, 4) for d, v in sorted(diff_avgs.items())},
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}", flush=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"GEMMA 4 E2B — French Cuisine Baseline Knowledge")
    print(f"{'='*60}")
    print(f"  Overall score:       {avg_score:.3f} / 1.000")
    print(f"  Verification pass:   {n_passed}/{len(results)} ({n_passed/len(results)*100:.1f}%)")
    print(f"\n  By category:")
    for c, v in sorted(cat_avgs.items()):
        n = len(scores_by_cat[c])
        print(f"    {c:30s}: {v:.3f}  (n={n})")
    print(f"\n  By difficulty:")
    for d, v in sorted(diff_avgs.items()):
        n = len(scores_by_diff[d])
        print(f"    {d:10s}: {v:.3f}  (n={n})")
    print(f"\n  Time: {elapsed:.0f}s ({len(results)/elapsed*60:.0f} prompts/min)")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    if avg_score > 0.80:
        print(f"  ** HIGH baseline ({avg_score:.3f}) — Gemma 4 E2B already knows French cuisine well.")
        print(f"  ** DD improvement ceiling is LOW. Consider a different domain or smaller base model.")
    elif avg_score > 0.60:
        print(f"  ** MODERATE baseline ({avg_score:.3f}) — room for DD improvement exists.")
        print(f"  ** Viable test case, but gains may be modest.")
    else:
        print(f"  ** LOW baseline ({avg_score:.3f}) — plenty of room for DD to show improvement.")
        print(f"  ** Good test case for disposition distillation.")
    print(flush=True)


if __name__ == "__main__":
    main()
