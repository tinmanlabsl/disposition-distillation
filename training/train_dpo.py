# Step 15: DPO on Type 3 judged counterexamples
# Per PLAN.md: load SFT adapter, 2-3 epochs, beta=0.1
import os, json, torch
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from unsloth import FastModel, PatchDPOTrainer
PatchDPOTrainer()
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

SFT_ADAPTER = '/workspace/persistent/dd-v2/checkpoints/chef_sft/final'
OUT_DIR     = '/workspace/persistent/dd-v2/checkpoints/chef_dpo'
DATA        = '/workspace/persistent/data/type3_counter_judged.jsonl'
MAX_LEN     = 2048

print('loading data...', flush=True)
rows = []
with open(DATA) as f:
    for l in f:
        r = json.loads(l)
        rows.append({'prompt': r['prompt'], 'chosen': r['chosen'], 'rejected': r['rejected']})
print(f'  pairs: {len(rows)}', flush=True)

print('loading SFT model (4bit + adapter)...', flush=True)
model, tok = FastModel.from_pretrained(SFT_ADAPTER, max_seq_length=MAX_LEN, load_in_4bit=True, full_finetuning=False)
print(f'  vram={torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)

# Adapter already loaded; enable lora params for training
model.train()
for n, pp in model.named_parameters():
    if 'lora_' in n:
        pp.requires_grad_(True)
trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  trainable={trn:,}', flush=True)

def fmt(ex):
    u = [{'role':'user','content':[{'type':'text','text': ex['prompt']}]}]
    prompt_text = tok.apply_chat_template(u, tokenize=False, add_generation_prompt=True)
    return {'prompt': prompt_text, 'chosen': ex['chosen'], 'rejected': ex['rejected']}

ds = Dataset.from_list(rows).map(fmt)
print(f'  dataset rows={len(ds)}', flush=True)

cfg = DPOConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=5e-6,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=5,
    save_strategy='steps',
    save_steps=50,
    save_total_limit=4,
    bf16=True,
    optim='adamw_8bit',
    beta=0.1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    max_length=1280,
    max_prompt_length=384,
    report_to='none',
    seed=42,
)
# Force text-only path: temporarily hide model_type from vision mapping
_orig_mt = model.config.model_type
model.config.model_type = 'gemma4_text'  # any non-vision type name
tok_text = tok.tokenizer if hasattr(tok, 'tokenizer') else tok
trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, processing_class=tok_text)
model.config.model_type = _orig_mt
print('starting DPO training...', flush=True)
trainer.train()
trainer.save_model(f'{OUT_DIR}/final')
tok.save_pretrained(f'{OUT_DIR}/final')
print('=== DPO DONE ===', flush=True)
