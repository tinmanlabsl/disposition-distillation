import os, json, time
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from unsloth import FastModel
import torch

ADAPTER = '/workspace/persistent/dd-v2/checkpoints/chef_dpo/final'
PROMPTS = [
    "Service is in 7 minutes. Two tables ordered Sole Meuniere a la minute. The clarified butter just smoked and turned brown-black; my mise en place has no more clarified butter and the sous chef is pulling stocks. The sole is portioned and dredged. What do I do RIGHT NOW, in order, to save the dish without restarting the butter?",
    "I am plating a Saint-Honore for a 12-top in 4 minutes. The chiboustine has weeped a thin layer of liquid at the base of the choux ring and the caramel cage has begun to sweat in the walk-in humidity. I cannot remake either component. Walk me through the exact recovery sequence, what to sacrifice, and what to swap.",
]

print('loading...', flush=True)
m, t = FastModel.from_pretrained(ADAPTER, max_seq_length=2048, load_in_4bit=True, full_finetuning=False)
FastModel.for_inference(m)
text_tok = t.tokenizer if hasattr(t, 'tokenizer') else t
text_tok.padding_side = 'left'
if text_tok.pad_token is None:
    text_tok.pad_token = text_tok.eos_token

for p in PROMPTS:
    msgs = [[{'role':'user','content':[{'type':'text','text':p}]}]]
    rendered = [t.apply_chat_template(msgs[0], add_generation_prompt=True, tokenize=False)]
    enc = text_tok(rendered, return_tensors='pt', padding=True, truncation=True, max_length=1280).to('cuda')
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=600, do_sample=False, use_cache=True, pad_token_id=text_tok.pad_token_id)
    nt = o[0][enc['input_ids'].shape[1]:]
    resp = text_tok.decode(nt, skip_special_tokens=True)
    print('=' * 70)
    print('Q:', p[:200])
    print('-' * 70)
    print(resp)
    print()
