#!/usr/bin/env python3
"""
Gemma 4 E2B LoRA sanity check.
Uses community monkey-patch for Gemma4ClippableLinear (peft#3129).
"""
import json, torch, torch.nn as nn, gc, subprocess, signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

RESULTS_DIR = "/workspace/persistent/eval"

# ── Monkey-patch ClippableLinear BEFORE model load ──────────────────────────
# Source: dev.to/dentity007, HF google/gemma-4-31B#3
from transformers.models.gemma4 import modeling_gemma4

class PatchedClippableLinear(nn.Linear):
    def __init__(self, config, in_features, out_features):
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, x):
        if self.use_clipped_linears:
            x = torch.clamp(x, self.input_min, self.input_max)
        out = nn.Linear.forward(self, x)
        if self.use_clipped_linears:
            out = torch.clamp(out, self.output_min, self.output_max)
        return out

modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear
print("  Patched Gemma4ClippableLinear -> nn.Linear subclass", flush=True)
# ────────────────────────────────────────────────────────────────────────────

def gpu_check(label):
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True)
    print(f"  GPU[{label}]: {r.stdout.strip()}", flush=True)

TRAIN_PAIRS = [
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

TEST_PROMPTS = [
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

print("Loading Gemma 4 E2B (with patch)...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

tok = AutoTokenizer.from_pretrained("/workspace/persistent/models/gemma-4-e2b", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/persistent/models/gemma-4-e2b",
    dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, attn_implementation="eager",
)

lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    exclude_modules=["vision_tower", "multi_modal_projector"],
    lora_dropout=0.05, task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
gpu_check("after_lora")

# Build training texts
train_texts = []
for p, r in TRAIN_PAIRS:
    msgs = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False)
    except Exception:
        text = f"User: {p}\nAssistant: {r}"
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

# Test generation
model.eval()
think_blocks = 0
garbage = 0
gen_results = []

for prompt in TEST_PROMPTS:
    msgs = [{"role": "user", "content": prompt}]
    try:
        input_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        input_text = f"User: {prompt}\nAssistant:"

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
    gen_results.append({"prompt": prompt, "response": resp[:300], "think": has_think, "coherent": is_coherent})
    print(f"  Q: {prompt[:50]}  A: {resp[:120]}  [think={has_think}]", flush=True)

verdict = think_blocks == 0 and garbage <= 1
print(f"\n  VERDICT: gemma-4-e2b — think={think_blocks}/10, garbage={garbage}/10 -> {'PASS' if verdict else 'FAIL'}", flush=True)

result = {
    "model": "gemma-4-e2b", "passed": verdict,
    "think_blocks": think_blocks, "garbage": garbage, "avg_loss": total_loss / 5,
    "generations": gen_results,
}
with open(f"{RESULTS_DIR}/gemma4_e2b_lora_final.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"  Saved: {RESULTS_DIR}/gemma4_e2b_lora_final.json", flush=True)
