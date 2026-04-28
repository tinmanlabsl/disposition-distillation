"""Phase B.1 attribution — Adversarial-self disposition on Gemma 4 E2B.

Cohort split: self_verification score from eval_judge_rubric.
  'advH' (high self-verif) : score >= 4  (cohort ~n=11)
  'advL' (low self-verif)  : score <= 2  (cohort ~n=69)

Reuses path_b_attribution.py's hook logic. Writes ranked head list to
  dd-v2/eval/results/path_b_attribution_gemma4_adversarial.json
"""
import os, json, time
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from unsloth import FastModel
import torch
import numpy as np

MODEL_ID = 'unsloth/gemma-4-E2B-it'
TAG = 'gemma4_adversarial'
LAST_N = 32

BASELINE = '/workspace/persistent/dd-v2/eval/results/baseline_gemma4_v2_from_step16.json'
RUBRIC   = '/workspace/persistent/dd-v2/eval/results/judge_rubric.json'
OUT = f'/workspace/persistent/dd-v2/eval/results/path_b_attribution_{TAG}.json'


def load_labeled_items():
    base = json.load(open(BASELINE, 'r', encoding='utf-8'))['items']
    scores = json.load(open(RUBRIC, 'r', encoding='utf-8'))['conditions']['baseline']['scores']
    assert len(base) == len(scores), f'len mismatch {len(base)} vs {len(scores)}'
    items = []
    for it, sc in zip(base, scores):
        if 'error' in sc: continue
        sv = sc.get('self_verification')
        if sv is None: continue
        if sv >= 4:   label = 'advH'
        elif sv <= 2: label = 'advL'
        else: continue
        items.append({
            'i': it['i'], 'prompt': it['prompt'], 'response': it['response'],
            'self_verif': sv, 'label': label,
        })
    return items


def main():
    print(f'=== Phase B.1 attribution (Adversarial-self): {MODEL_ID} ===', flush=True)
    print('loading model...', flush=True)
    model, tok = FastModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=2048, dtype=torch.bfloat16,
        load_in_4bit=False, trust_remote_code=True,
    )
    FastModel.for_inference(model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model.eval()

    decoder_list = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith('.layers') and len(mod) >= 4:
            if hasattr(mod[0], 'self_attn') and hasattr(mod[0].self_attn, 'o_proj'):
                decoder_list = mod; break
    assert decoder_list is not None
    attn_layer_refs = [(i, l) for i, l in enumerate(decoder_list)
                       if hasattr(l, 'self_attn') and hasattr(l.self_attn, 'o_proj')]
    n_layers = len(attn_layer_refs)
    o0 = attn_layer_refs[0][1].self_attn.o_proj
    o_in_dim = o0.weight.shape[1]
    n_heads = model.config.text_config.num_attention_heads
    head_dim = o_in_dim // n_heads
    print(f'arch: layers={n_layers} heads={n_heads} head_dim={head_dim}', flush=True)

    captured = {}
    hooks = []
    def make_hook(L):
        def hook(module, inputs, output):
            captured[L] = inputs[0].detach()
        return hook
    for L, (_idx, layer) in enumerate(attn_layer_refs):
        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_hook(L)))

    items = load_labeled_items()
    buckets = {'advH': [], 'advL': []}
    for it in items: buckets[it['label']].append(it)
    print(f'cohorts: advH={len(buckets["advH"])} advL={len(buckets["advL"])}', flush=True)

    per_norms = {'advH': [], 'advL': []}
    t0 = time.time(); processed = 0
    total = sum(len(v) for v in buckets.values())
    for label, lst in buckets.items():
        for it in lst:
            try:
                msgs = [{'role':'user','content': it['prompt']}]
                prompt_rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            except Exception:
                prompt_rendered = it['prompt']
            full = prompt_rendered + it['response']
            prompt_ids = tok(text=prompt_rendered, return_tensors='pt', truncation=True, max_length=2048)['input_ids']
            full_ids   = tok(text=full,            return_tensors='pt', truncation=True, max_length=2048)['input_ids']
            prompt_len = prompt_ids.shape[1]
            total_len  = full_ids.shape[1]
            if total_len <= prompt_len + 2: continue
            resp_start = max(prompt_len, total_len - LAST_N)
            enc = full_ids.to('cuda')
            with torch.no_grad():
                _ = model(input_ids=enc, attention_mask=torch.ones_like(enc), use_cache=False)
            per_layer = np.zeros((n_layers, n_heads), dtype=np.float32)
            for L in range(n_layers):
                act = captured[L]
                slice_ = act[0, resp_start:total_len, :] if act.dim() == 3 else act[resp_start:total_len, :]
                D = slice_.shape[-1]; hd = D // n_heads
                slice_ = slice_.reshape(slice_.shape[0], n_heads, hd)
                per_layer[L] = slice_.norm(dim=-1).mean(dim=0).float().cpu().numpy()
            per_norms[label].append(per_layer)
            captured.clear()
            processed += 1
            if processed % 10 == 0:
                print(f'  [{processed}/{total}]', flush=True)

    for h in hooks: h.remove()

    advH = np.stack(per_norms['advH']) if per_norms['advH'] else np.zeros((0,n_layers,n_heads),dtype=np.float32)
    advL = np.stack(per_norms['advL']) if per_norms['advL'] else np.zeros((0,n_layers,n_heads),dtype=np.float32)
    mH = advH.mean(0); mL = advL.mean(0)
    # Overconfidence direction: advH - advL (heads firing more when SELF-VERIFICATION is HIGH)
    delta = mH - mL
    pooled = np.sqrt(((advH.shape[0]-1)*advH.var(0,ddof=1) + (advL.shape[0]-1)*advL.var(0,ddof=1))
                     / max(advH.shape[0]+advL.shape[0]-2, 1))
    d = np.where(pooled > 1e-8, delta / pooled, 0.0)

    orig_indices = [i for i,_ in attn_layer_refs]
    flat = []
    for L in range(n_layers):
        for H in range(n_heads):
            flat.append({'layer': orig_indices[L], 'attn_layer_idx': L, 'head': H,
                         'mean_advH': float(mH[L,H]), 'mean_advL': float(mL[L,H]),
                         'delta': float(delta[L,H]), 'cohen_d': float(d[L,H])})
    # Sort by ABS(d) for adversarial — we care about either direction (heads that differ)
    flat_abs = sorted(flat, key=lambda r: abs(r['cohen_d']), reverse=True)
    flat_pos = sorted(flat, key=lambda r: r['cohen_d'], reverse=True)
    result = {
        'model_id': MODEL_ID, 'tag': TAG,
        'cohorts': {'advH': len(buckets['advH']), 'advL': len(buckets['advL'])},
        'top20_by_abs_d': flat_abs[:20],
        'top20_overconf_heads': flat_pos[:20],  # keep key name so B.2 script can read it
        'bottom20_by_cohen_d': flat_pos[-20:],
        'all_heads_ranked_by_cohen_d': flat_pos,
    }
    json.dump(result, open(OUT, 'w'), indent=1)
    print('\n=== TOP 10 by |Cohen d| ===', flush=True)
    for r in flat_abs[:10]:
        print(f"  L{r['layer']:2d} H{r['head']:2d}  d={r['cohen_d']:+.3f}  "
              f"H={r['mean_advH']:.3f}  L={r['mean_advL']:.3f}", flush=True)
    print(f'\nsaved -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
