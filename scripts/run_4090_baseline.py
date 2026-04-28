"""4090: Qwen3.5 full HumanEval + LoRA, Gemma 4 E2B LoRA rerun"""
import json, time, re, torch, traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

CACHE = "/workspace/persistent/eval/humaneval_problems.json"
with open(CACHE) as f:
    problems = json.load(f)

def extract_code(resp):
    text = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    m = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    return m.group(1) if m else text

def check_solution(prob, completion):
    code = prob["prompt"] + completion + "\n" + prob["test"] + "\ncheck(" + prob["entry_point"] + ")"
    try:
        exec(code, {})
        return True
    except:
        return False

def run_humaneval(model_name, model_path):
    print(f"\n{'='*60}")
    print(f"HumanEval: {model_name}")
    print(f"{'='*60}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, attn_implementation="sdpa"
    )
    model.eval()

    results = []
    passed = 0
    start = time.time()

    for i, p in enumerate(problems):
        prompt_text = "Complete the following Python function. Output ONLY the function body code, no explanation.\n\n" + p["prompt"]
        messages = [{"role": "user", "content": prompt_text}]
        try:
            input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            try:
                input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = prompt_text

        ids = tok(input_text, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=1024, do_sample=False, pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        comp = extract_code(resp)
        ok = check_solution(p, comp)
        if ok:
            passed += 1
        results.append({"task_id": p["task_id"], "passed": ok, "has_think": "<think>" in resp, "resp_len": len(resp), "raw": resp[:300]})

        if (i + 1) % 20 == 0:
            el = time.time() - start
            print(f"  [{i+1}/164] {passed}/{i+1} ({passed/(i+1)*100:.1f}%) | {(i+1)/el*60:.0f} prob/min", flush=True)

    el = time.time() - start
    score = passed / len(problems) * 100
    think_ct = sum(1 for r in results if r["has_think"])
    print(f"\nRESULT: {model_name}  Pass@1: {passed}/{len(problems)} = {score:.1f}%  Think: {think_ct}  Time: {el:.0f}s", flush=True)

    del model, tok
    torch.cuda.empty_cache()
    return {"model": model_name, "score": score, "passed": passed, "total": len(problems), "think_blocks": think_ct, "elapsed": el, "results": results}

def run_lora_sanity(model_name, model_path):
    print(f"\n{'='*60}")
    print(f"LoRA Sanity: {model_name}")
    print(f"{'='*60}", flush=True)

    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, attn_implementation="sdpa"
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    print(f"  Target modules: {target_modules}", flush=True)

    lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=target_modules, lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    pairs = [
        ("What is a bechamel sauce?", "Bechamel is a white sauce made from butter, flour, and milk."),
        ("How do you make a roux?", "A roux is made by cooking equal parts flour and fat together."),
        ("What region is bouillabaisse from?", "Bouillabaisse is a traditional fish stew from Marseille, Provence."),
        ("Explain the julienne cut.", "Julienne is a knife cut producing thin strips, approximately 3mm x 3mm x 50mm."),
        ("What temperature for caramel?", "Caramel forms at approximately 160-177 degrees Celsius."),
        ("What are the five mother sauces?", "The five French mother sauces are bechamel, veloute, espagnole, hollandaise, and tomato."),
        ("What is a mirepoix?", "Mirepoix is diced onions, carrots, and celery in a 2:1:1 ratio."),
        ("How do you deglaze a pan?", "Add liquid to a hot pan after searing to dissolve the fond from the bottom."),
        ("What is confit?", "Confit is meat slowly cooked in its own fat at low temperature."),
        ("What region is choucroute from?", "Choucroute garnie is from Alsace, featuring sauerkraut with sausages and pork."),
    ] * 5

    train_texts = []
    for p, r in pairs:
        msgs = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        except TypeError:
            try:
                text = tok.apply_chat_template(msgs, tokenize=False)
            except Exception:
                text = "User: " + p + "\nAssistant: " + r
        train_texts.append(text)

    encodings = tok(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    optimizer = AdamW(model.parameters(), lr=2e-4)
    model.train()

    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)

    total_loss = 0
    for batch_start in range(0, len(train_texts), 10):
        batch_ids = input_ids[batch_start:batch_start + 10]
        batch_mask = attention_mask[batch_start:batch_start + 10]
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        print(f"  Batch {batch_start // 10 + 1}/5, loss={loss.item():.4f}", flush=True)

    model.eval()
    test_prompts = [
        "What are the five French mother sauces?",
        "How do you make hollandaise?",
        "What is the difference between saute and braise?",
        "What region is cassoulet from?",
        "What temperature should oil be for deep frying?",
        "Explain the technique of deglazing.",
        "What is a bouquet garni?",
        "How do you clarify butter?",
        "What knife cut is brunoise?",
        "Name three dishes from Burgundy.",
    ]

    think_blocks = 0
    garbage = 0

    for prompt in test_prompts:
        msgs = [{"role": "user", "content": prompt}]
        try:
            input_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            try:
                input_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = "User: " + prompt + "\nAssistant:"

        ids = tok(input_text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        has_think = "<think>" in resp or "</think>" in resp
        is_coherent = len(resp.strip()) > 10 and any(c.isalpha() for c in resp)
        if has_think:
            think_blocks += 1
        if not is_coherent:
            garbage += 1
        print(f"  Q: {prompt[:50]}  A: {resp[:120]}  [think={has_think}]", flush=True)

    verdict = think_blocks == 0 and garbage <= 1
    print(f"\nLoRA RESULT: {model_name}  think={think_blocks}/10  garbage={garbage}/10  VERDICT={'PASS' if verdict else 'FAIL'}", flush=True)

    del model, tok
    torch.cuda.empty_cache()
    return {"model": model_name, "passed": verdict, "think_blocks": think_blocks, "garbage": garbage, "avg_loss": total_loss / 5}


if __name__ == "__main__":
    all_results = {}

    # 1. Qwen3.5 full HumanEval
    try:
        all_results["qwen35_humaneval"] = run_humaneval("qwen3.5-0.8b", "/workspace/persistent/models/qwen3.5-0.8b")
    except Exception as e:
        print(f"ERROR qwen3.5 humaneval: {e}")
        traceback.print_exc()
        all_results["qwen35_humaneval"] = {"error": str(e)}

    # 2. Qwen3.5 LoRA sanity
    try:
        all_results["qwen35_lora"] = run_lora_sanity("qwen3.5-0.8b", "/workspace/persistent/models/qwen3.5-0.8b")
    except Exception as e:
        print(f"ERROR qwen3.5 lora: {e}")
        traceback.print_exc()
        all_results["qwen35_lora"] = {"error": str(e)}

    # 3. Gemma 4 E2B LoRA rerun (explicit target_modules fix)
    try:
        all_results["gemma4_e2b_lora"] = run_lora_sanity("gemma-4-e2b", "/workspace/persistent/models/gemma-4-e2b")
    except Exception as e:
        print(f"ERROR gemma4 lora: {e}")
        traceback.print_exc()
        all_results["gemma4_e2b_lora"] = {"error": str(e)}

    with open("/workspace/persistent/eval/baseline_4090_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL DONE - saved to /workspace/persistent/eval/baseline_4090_results.json")
    print("=" * 60)
