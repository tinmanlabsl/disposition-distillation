# Step 16 ablation SFT — generic single-source SFT
import os, json, sys, torch, argparse
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from unsloth import FastModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--out',  required=True)
args = ap.parse_args()

MODEL = 'unsloth/gemma-4-E2B-it'
MAX_LEN = 2048

print(f'data={args.data}  out={args.out}', flush=True)
with open(args.data) as f:
    rows = [json.loads(l) for l in f if l.strip()]
print(f'  rows: {len(rows)}', flush=True)

model, tok = FastModel.from_pretrained(MODEL, max_seq_length=MAX_LEN, load_in_4bit=True, full_finetuning=False)
model = FastModel.get_peft_model(
    model,
    finetune_language_layers=True, finetune_vision_layers=False,
    finetune_attention_modules=True, finetune_mlp_modules=False,
    r=64, lora_alpha=128, lora_dropout=0, bias='none', random_state=42,
)

def fmt(ex):
    msgs = [
        {'role':'user','content':[{'type':'text','text':ex['prompt']}]},
        {'role':'assistant','content':[{'type':'text','text':ex['response']}]},
    ]
    return {'text': tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

ds = Dataset.from_list(rows).map(fmt, remove_columns=['prompt','response'])
print(f'  dataset rows={len(ds)}', flush=True)

cfg = SFTConfig(
    output_dir=args.out,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=5,
    save_strategy='no',
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
trainer.train()
trainer.save_model(f'{args.out}/final')
tok.save_pretrained(f'{args.out}/final')
print('=== DONE ===', flush=True)
