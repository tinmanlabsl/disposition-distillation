# Step 14: SFT on Types 1 + 2-clean + 4 — French Chef specialist
# Per PLAN.md: attn-only LoRA r=64 alpha=128, 3 epochs cosine, Gemma 4 E2B
import os, json, sys, torch
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'

from unsloth import FastModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

MODEL      = 'unsloth/gemma-4-E2B-it'
MAX_LEN    = 2048
OUT_DIR    = '/workspace/persistent/dd-v2/checkpoints/chef_sft'
DATA_DIR   = '/workspace/persistent/data'
T1 = f'{DATA_DIR}/type1_gold.jsonl'
T2 = f'{DATA_DIR}/type2_failure_clean.jsonl'
T4 = f'{DATA_DIR}/type4_boundary.jsonl'

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

print('loading data...', flush=True)
t1 = load_jsonl(T1); t2 = load_jsonl(T2); t4 = load_jsonl(T4)
print(f'  type1 gold:       {len(t1)}', flush=True)
print(f'  type2 clean:      {len(t2)}', flush=True)
print(f'  type4 boundary:   {len(t4)}', flush=True)

records = []
for r in t1:
    records.append({'prompt': r['prompt'], 'response': r['response']})
for r in t2:
    # single-turn: teach corrected response directly
    records.append({'prompt': r['prompt'], 'response': r['corrected_response']})
for r in t4:
    records.append({'prompt': r['prompt'], 'response': r['response']})
print(f'  TOTAL:            {len(records)}', flush=True)

print('loading model (4bit)...', flush=True)
model, tok = FastModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_LEN,
    load_in_4bit=True,
    full_finetuning=False,
)
print(f'  vram={torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

print('applying LoRA (attn-only, r=64, alpha=128)...', flush=True)
model = FastModel.get_peft_model(
    model,
    finetune_language_layers=True,
    finetune_vision_layers=False,
    finetune_attention_modules=True,
    finetune_mlp_modules=False,
    r=64, lora_alpha=128, lora_dropout=0, bias='none',
    random_state=42,
)
trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  trainable={trn:,}', flush=True)

def fmt(ex):
    msgs = [
        {'role': 'user',      'content': [{'type':'text','text': ex['prompt']}]},
        {'role': 'assistant', 'content': [{'type':'text','text': ex['response']}]},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {'text': text}

ds = Dataset.from_list(records).map(fmt, remove_columns=['prompt','response'])
print(f'  dataset rows={len(ds)}', flush=True)
print('sample[0]:', ds[0]['text'][:300], flush=True)

cfg = SFTConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # effective batch 32
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=5,
    save_strategy='steps',
    save_steps=50,
    save_total_limit=6,
    bf16=True,
    optim='adamw_8bit',
    weight_decay=0.01,
    max_length=MAX_LEN,
    dataset_text_field='text',
    packing=False,
    report_to='none',
    seed=42,
)

trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)
print('starting training...', flush=True)
trainer.train()
print('saving final adapter...', flush=True)
trainer.save_model(f'{OUT_DIR}/final')
tok.save_pretrained(f'{OUT_DIR}/final')
print('=== DONE ===', flush=True)
