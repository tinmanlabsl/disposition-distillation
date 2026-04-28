#!/usr/bin/env python3
"""DD v2 Step 11a — Generate Gemma 4 E2B Student Responses

Runs the BASE model (before any DD training) on all 684 cuisine prompts.
These are the "student" responses that will be compared against gold (Step 10)
to identify errors. Errors become Type 2 failure traces.

This script uses GPU only (no API calls) — runs in parallel with the
Type 1 API pipeline.

Saves each response immediately + checkpoints every 10 items.
"""
import json
import os
import signal
import sys
import time

signal.signal(signal.SIGHUP, signal.SIG_IGN)

MODEL_PATH = "/workspace/persistent/models/gemma-4-e2b"
PROMPTS_PATH = "/workspace/persistent/cuisine_test/chef_prompts.jsonl"
OUTPUT_PATH = "/workspace/persistent/data/student_responses.jsonl"
LOG_PATH = "/workspace/persistent/logs/step11a_student.log"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load prompts
    prompts = []
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    log(f"Loaded {len(prompts)} prompts")

    # Load existing outputs to resume
    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_ids.add(json.loads(line)["id"])
    log(f"Already completed: {len(completed_ids)}. Remaining: {len(prompts) - len(completed_ids)}")

    remaining = [p for p in prompts if p["id"] not in completed_ids]
    if not remaining:
        log("All prompts already processed!")
        return

    # Load model
    log(f"Loading Gemma 4 E2B from {MODEL_PATH}...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    log("Model loaded")

    completed = len(completed_ids)
    total = len(prompts)
    start = time.time()

    for i, p in enumerate(remaining):
        prompt_text = p["prompt"]

        # Simple direct prompt — no system message, just the question
        # This captures the base model's natural response style
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tok([input_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=1024,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

        result = {
            "id": p["id"],
            "prompt": prompt_text,
            "category": p.get("category", ""),
            "difficulty": p.get("difficulty", ""),
            "response": response,
            "response_length": len(response),
        }

        # Save immediately
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        completed += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            log(f"[{completed}/{total}] {rate:.1f} prompts/min | "
                f"Last: {p['id']} ({p['category']}/{p['difficulty']})")

    elapsed = time.time() - start
    log(f"\nDone. {completed}/{total} student responses. Time: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
