"""
Alpha-sweep contrastive decoding: logits = base + alpha*(dd - base).
Uses one model with LoRA adapter toggled on/off (disable_adapter) so
base and DD share weights except for LoRA deltas. Maintains two separate
KV caches across the decode loop.

Instruments per-token: base top1 prob, top1/top2 margin, KL(base || mixed).
Outputs JSON for downstream v3 checklist scoring.
"""
import os, json, time, math, sys
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from pathlib import Path
from unsloth import FastModel
import torch
import torch.nn.functional as F

ADAPTER   = '/workspace/persistent/dd-v2/checkpoints/chef_dpo/final'
GOLD_PATH = '/workspace/persistent/dd-v2/data/type1_gold.jsonl'
OUT_PATH  = '/workspace/persistent/dd-v2/eval/results/alpha_sweep.json'
ALPHAS    = [0.0, 0.25, 0.5, 1.0, 2.0]
MAX_NEW   = 400
# Gold indices to probe (eval prompts with known canonical checklists)
GOLD_IDX  = [3, 5, 9, 11, 17, 23, 29, 52, 59, 83]

NOVEL_PROMPTS = [
    ("novel_sole", "Service is in 7 minutes. Two tables ordered Sole Meuniere a la minute. The clarified butter just smoked and turned brown-black; my mise en place has no more clarified butter and the sous chef is pulling stocks. The sole is portioned and dredged. What do I do RIGHT NOW, in order, to save the dish without restarting the butter?"),
    ("novel_sthonore", "I am plating a Saint-Honore for a 12-top in 4 minutes. The chiboustine has weeped a thin layer of liquid at the base of the choux ring and the caramel cage has begun to sweat in the walk-in humidity. I cannot remake either component. Walk me through the exact recovery sequence, what to sacrifice, and what to swap."),
]

def load_gold_prompts(indices):
    want = set(indices)
    out = {}
    with open(GOLD_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i in want:
                r = json.loads(line)
                out[i] = r.get('prompt') or r.get('question') or r.get('instruction')
            if len(out) == len(want):
                break
    return [(f'gold_{i}', out[i]) for i in indices if i in out]

def render(tok, prompt):
    msgs = [{'role':'user','content':[{'type':'text','text':prompt}]}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

@torch.no_grad()
def sweep_one(model, tok, text_tok, prompt_text, alpha):
    """Greedy decode with mixed logits across two adapter states. Returns (text, telemetry)."""
    rendered = render(tok, prompt_text)
    enc = text_tok(rendered, return_tensors='pt').to('cuda')
    input_ids = enc['input_ids']
    attn = enc['attention_mask']

    # Prefill BASE (adapter disabled)
    with model.disable_adapter():
        out_b = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    pkv_b = out_b.past_key_values
    base_logits = out_b.logits[:, -1, :].float()

    # Prefill DD (adapter enabled)
    out_d = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    pkv_d = out_d.past_key_values
    dd_logits = out_d.logits[:, -1, :].float()

    eos = text_tok.eos_token_id
    eot = None
    try:
        eot = text_tok.convert_tokens_to_ids('<end_of_turn>')
        if eot == text_tok.unk_token_id:
            eot = None
    except Exception:
        pass

    generated = []
    telemetry = []

    cur_attn = attn
    for step in range(MAX_NEW):
        base_lp = F.log_softmax(base_logits, dim=-1)
        mixed = base_logits + alpha * (dd_logits - base_logits)
        mixed_lp = F.log_softmax(mixed, dim=-1)

        # Telemetry from base
        base_p = base_lp.exp()
        top2 = torch.topk(base_p, 2, dim=-1)
        top1_p = top2.values[0, 0].item()
        top2_p = top2.values[0, 1].item()
        margin = top1_p - top2_p
        kl = F.kl_div(mixed_lp, base_lp, reduction='sum', log_target=True).item()

        next_tok = torch.argmax(mixed, dim=-1, keepdim=True)  # [1,1]
        tok_id = next_tok.item()
        generated.append(tok_id)
        telemetry.append({'t': step, 'base_top1': round(top1_p, 4), 'margin': round(margin, 4), 'kl': round(kl, 4), 'tok': tok_id})

        if tok_id == eos or (eot is not None and tok_id == eot):
            break

        cur_attn = torch.cat([cur_attn, torch.ones((1,1), dtype=cur_attn.dtype, device=cur_attn.device)], dim=1)

        with model.disable_adapter():
            out_b = model(input_ids=next_tok, attention_mask=cur_attn, past_key_values=pkv_b, use_cache=True)
        pkv_b = out_b.past_key_values
        base_logits = out_b.logits[:, -1, :].float()

        out_d = model(input_ids=next_tok, attention_mask=cur_attn, past_key_values=pkv_d, use_cache=True)
        pkv_d = out_d.past_key_values
        dd_logits = out_d.logits[:, -1, :].float()

    text = text_tok.decode(generated, skip_special_tokens=True)
    return text, telemetry


def main():
    print('loading model+adapter...', flush=True)
    m, t = FastModel.from_pretrained(ADAPTER, max_seq_length=2048, load_in_4bit=True, full_finetuning=False)
    FastModel.for_inference(m)
    text_tok = t.tokenizer if hasattr(t, 'tokenizer') else t
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    prompts = list(NOVEL_PROMPTS) + load_gold_prompts(GOLD_IDX)
    print(f'loaded {len(prompts)} prompts, alphas={ALPHAS}', flush=True)

    results = {'alphas': ALPHAS, 'items': []}
    t0 = time.time()
    for tag, p in prompts:
        item = {'tag': tag, 'prompt': p, 'by_alpha': {}}
        for a in ALPHAS:
            ts = time.time()
            try:
                text, tele = sweep_one(m, t, text_tok, p, a)
            except Exception as e:
                text, tele = f'[ERROR {type(e).__name__}: {e}]', []
            dt = time.time() - ts
            item['by_alpha'][str(a)] = {'text': text, 'telemetry': tele, 'secs': round(dt, 1), 'n_tok': len(tele)}
            print(f'[{tag} a={a}] {len(tele)}tok in {dt:.1f}s', flush=True)
            # Save incrementally
            with open(OUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(results | {'items': results['items'] + [item]}, f, indent=1, ensure_ascii=False)
        results['items'].append(item)
        with open(OUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=1, ensure_ascii=False)

    print(f'DONE {time.time()-t0:.0f}s -> {OUT_PATH}', flush=True)

if __name__ == '__main__':
    main()
