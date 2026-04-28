"""Phase B.1 — attention-head attribution for overconfidence behavior.

Usage:
  python path_b_attribution.py <model_id> <tag>

  Tags:
    gemma4   → reuses dd-v2/eval/results/baseline_gemma4_v2_from_step16.json
    qwen35_0.8b → reuses dd-v2/eval/results/baseline_qwen35_0.8b_v2_disposition_profile.json

For each labeled sample (cr / cw / hr / hw), forward-pass the base model over
`prompt + response` and extract the per-head attention-output activation norm
at the last N response tokens. Rank heads by mean(cw) - mean(cr).

This is a forward-only mechanistic probe. No training, no gradients, no
weight modification.
"""
import os, sys, json, time
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from unsloth import FastModel
import torch
import numpy as np

MODEL_ID = sys.argv[1]
TAG      = sys.argv[2]
LAST_N   = 32  # last 32 response tokens — where delivery/commitment is expressed

EVAL_BASELINE = '/workspace/persistent/dd-v2/eval/results/eval_baseline.json'
RESULTS_DIR   = '/workspace/persistent/dd-v2/eval/results'
OUT = f'{RESULTS_DIR}/path_b_attribution_{TAG}.json'

# Different baseline files store the same 2x2 data in slightly different shapes.
PROFILE_FILES = {
    'gemma4':      f'{RESULTS_DIR}/baseline_gemma4_v2_from_step16.json',
    'smollm':      f'{RESULTS_DIR}/baseline_smollm_v2_disposition_profile.json',
    'qwen35_0.8b': f'{RESULTS_DIR}/baseline_qwen35_0.8b_v2_disposition_profile.json',
}


def load_labeled_items():
    """Return list of dicts with keys: prompt, response, correct, verbal, label."""
    pf = PROFILE_FILES[TAG]
    data = json.load(open(pf, 'r', encoding='utf-8'))
    rows = json.load(open(EVAL_BASELINE, 'r', encoding='utf-8'))['rows']

    items = []
    if TAG == 'gemma4':
        # {items: [{i, prompt, response, coverage, correct, verbal}]}
        for it in data['items']:
            items.append({
                'i': it['i'],
                'prompt': it['prompt'],
                'response': it['response'],
                'correct': bool(it['correct']),
                'verbal': it['verbal'],
            })
    else:
        # baseline_crossmodel.py format:
        # {summary, samples: [{prompt_i, topic, coverage, correct, text, verbal}]}
        for s in data['samples']:
            i = s['prompt_i']
            prompt = rows[i]['prompt'] if i < len(rows) else None
            if prompt is None:
                continue
            items.append({
                'i': i,
                'prompt': prompt,
                'response': s['text'],
                'correct': bool(s['correct']),
                'verbal': s['verbal'],
            })

    for it in items:
        if it['correct'] and it['verbal'] == 'assert': it['label'] = 'cr'
        elif it['correct'] and it['verbal'] == 'hedge': it['label'] = 'hr'
        elif (not it['correct']) and it['verbal'] == 'assert': it['label'] = 'cw'
        else: it['label'] = 'hw'
    return items


def main():
    print(f'=== Phase B.1 attribution: {MODEL_ID} (tag {TAG}) ===', flush=True)
    print('loading model...', flush=True)
    model, tok = FastModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        trust_remote_code=True,
        attn_implementation='sdpa' if 'qwen3.5' not in MODEL_ID.lower() and 'qwen3_5' not in MODEL_ID.lower() else None,
    )
    FastModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()

    # Discover model architecture. Multimodal configs (Gemma 4) nest text config.
    cfg = model.config
    text_cfg = getattr(cfg, 'text_config', None) or cfg
    n_layers = getattr(text_cfg, 'num_hidden_layers', None) or getattr(text_cfg, 'n_layer', None)
    n_heads  = getattr(text_cfg, 'num_attention_heads', None)
    hidden   = getattr(text_cfg, 'hidden_size', None) or getattr(text_cfg, 'd_model', None)
    head_dim = getattr(text_cfg, 'head_dim', None) or (hidden // n_heads)
    print(f'arch: layers={n_layers} heads={n_heads} hidden={hidden} head_dim={head_dim}', flush=True)

    # Locate layers with standard self_attn.o_proj. For hybrid models (Qwen3.5
    # has DeltaNet + attention), only the true attention layers are collected.
    attn_layer_refs = []  # list of (orig_layer_idx, module_with_o_proj)
    # Find the top-level ModuleList of decoder layers
    decoder_list = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 4:
            # heuristic: prefer a ModuleList under '*.layers'
            if name.endswith('.layers'):
                decoder_list = mod
                print(f'found decoder list at: {name} (count={len(mod)})', flush=True)
                break
    assert decoder_list is not None, 'could not find decoder layers ModuleList'
    for idx, layer in enumerate(decoder_list):
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            attn_layer_refs.append((idx, layer))
    assert len(attn_layer_refs) > 0, 'no attention layers with self_attn.o_proj'
    print(f'attention layers: {len(attn_layer_refs)} / {len(decoder_list)} total '
          f'(indices: {[i for i,_ in attn_layer_refs]})', flush=True)
    n_layers = len(attn_layer_refs)

    # Derive n_heads / head_dim from the first attention layer's o_proj shape.
    o0 = attn_layer_refs[0][1].self_attn.o_proj
    o_in_dim = o0.weight.shape[1]  # = n_heads * head_dim
    if head_dim is None or head_dim == 0:
        head_dim = o_in_dim // n_heads
    else:
        n_heads = o_in_dim // head_dim
    print(f'resolved: n_attn_layers={n_layers} n_heads={n_heads} head_dim={head_dim} o_in_dim={o_in_dim}', flush=True)

    # Hook o_proj inputs on every layer. o_proj takes [B, T, n_heads*head_dim]
    # which is the concatenation of per-head outputs BEFORE mixing — this is exactly
    # where per-head activation norms can be read.
    captured = {}  # attn_layer_idx (0..n_layers-1) -> tensor
    hooks = []
    orig_layer_indices = [i for i, _ in attn_layer_refs]
    def make_hook(L):
        def hook(module, inputs, output):
            captured[L] = inputs[0].detach()
        return hook
    for L, (_orig_idx, layer) in enumerate(attn_layer_refs):
        o_proj = layer.self_attn.o_proj
        hooks.append(o_proj.register_forward_hook(make_hook(L)))

    items = load_labeled_items()
    buckets = {'cr': [], 'cw': [], 'hr': [], 'hw': []}
    for it in items:
        buckets[it['label']].append(it)
    print(f'cohorts: cr={len(buckets["cr"])} hr={len(buckets["hr"])} cw={len(buckets["cw"])} hw={len(buckets["hw"])}', flush=True)

    # For each sample: tokenize prompt and prompt+response, forward, grab per-head norm
    # over last LAST_N response tokens.
    per_item_norms = {}  # label -> list of np.array [n_layers, n_heads]
    for b in buckets:
        per_item_norms[b] = []

    t0 = time.time()
    total = sum(len(v) for v in buckets.values())
    processed = 0
    for label, lst in buckets.items():
        for it in lst:
            try:
                msgs = [{'role':'user','content': it['prompt']}]
                prompt_rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            except Exception:
                prompt_rendered = it['prompt']
            # Full sequence = prompt + response text. We need prompt_len to know where
            # the response starts so we can slice out the last LAST_N response tokens.
            full = prompt_rendered + it['response']
            prompt_ids = tok(text=prompt_rendered, return_tensors='pt', truncation=True, max_length=2048)['input_ids']
            full_ids   = tok(text=full,            return_tensors='pt', truncation=True, max_length=2048)['input_ids']
            prompt_len = prompt_ids.shape[1]
            total_len  = full_ids.shape[1]
            if total_len <= prompt_len + 2:
                continue  # empty response
            resp_end = total_len
            resp_start = max(prompt_len, resp_end - LAST_N)

            enc = full_ids.to('cuda')
            attn_mask = torch.ones_like(enc)
            with torch.no_grad():
                _ = model(input_ids=enc, attention_mask=attn_mask, use_cache=False)

            # Infer true per-head dim from first captured tensor
            if processed == 0:
                sample_act = captured[0]
                # [1, T, D] or [T, D] depending on module
                D_actual = sample_act.shape[-1]
                head_dim_actual = D_actual // n_heads
                print(f'runtime D={D_actual} per head={head_dim_actual} (config head_dim={head_dim})', flush=True)
                _head_dim_runtime = head_dim_actual
            # Aggregate per-head norms over the last N response tokens
            per_layer = np.zeros((n_layers, n_heads), dtype=np.float32)
            for L in range(n_layers):
                act = captured[L]
                if act.dim() == 3:
                    slice_ = act[0, resp_start:resp_end, :]
                else:
                    slice_ = act[resp_start:resp_end, :]
                D = slice_.shape[-1]
                hd = D // n_heads
                slice_ = slice_.reshape(slice_.shape[0], n_heads, hd)
                head_norms = slice_.norm(dim=-1).mean(dim=0).float().cpu().numpy()  # [n_heads]
                per_layer[L] = head_norms
            per_item_norms[label].append(per_layer)
            captured.clear()

            processed += 1
            if processed % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / processed * (total - processed)
                print(f'  [{processed}/{total}] {label} elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

    for h in hooks:
        h.remove()

    # Aggregate
    def stack(label):
        xs = per_item_norms[label]
        return np.stack(xs) if xs else np.zeros((0, n_layers, n_heads), dtype=np.float32)

    cr = stack('cr')  # [n_cr, n_layers, n_heads]
    cw = stack('cw')
    hr = stack('hr')
    hw = stack('hw')

    mean_cr = cr.mean(0) if cr.size else None
    mean_cw = cw.mean(0) if cw.size else None
    mean_hr = hr.mean(0) if hr.size else None
    mean_hw = hw.mean(0) if hw.size else None

    # Overconfidence score per head
    overconf = mean_cw - mean_cr  # [n_layers, n_heads]

    # Pooled std for effect size
    if cr.size and cw.size:
        pooled_std = np.sqrt(
            ((cr.shape[0]-1) * cr.var(0, ddof=1) + (cw.shape[0]-1) * cw.var(0, ddof=1))
            / max(cr.shape[0] + cw.shape[0] - 2, 1)
        )
        cohen_d = np.where(pooled_std > 1e-8, overconf / pooled_std, 0.0)
    else:
        cohen_d = np.zeros_like(overconf)

    # Rank
    flat = []
    for L in range(n_layers):
        orig_L = orig_layer_indices[L]
        for H in range(n_heads):
            flat.append({
                'layer': orig_L, 'attn_layer_idx': L, 'head': H,
                'mean_cw': float(mean_cw[L, H]) if mean_cw is not None else None,
                'mean_cr': float(mean_cr[L, H]) if mean_cr is not None else None,
                'overconf_score': float(overconf[L, H]),
                'cohen_d': float(cohen_d[L, H]),
            })
    flat.sort(key=lambda r: r['cohen_d'], reverse=True)

    result = {
        'model_id': MODEL_ID,
        'tag': TAG,
        'arch': {'n_attn_layers': n_layers, 'n_heads': n_heads, 'head_dim': head_dim,
                 'orig_layer_indices': orig_layer_indices},
        'cohorts': {k: len(v) for k, v in buckets.items()},
        'last_n_tokens': LAST_N,
        'top20_overconf_heads': flat[:20],
        'bottom5_overconf_heads': flat[-5:],
        'all_heads_ranked_by_cohen_d': flat,
    }
    json.dump(result, open(OUT, 'w'), indent=1)

    print('\n=== TOP 10 overconfidence heads (by Cohen\'s d) ===', flush=True)
    for r in flat[:10]:
        print(f"  L{r['layer']:2d} H{r['head']:2d}  d={r['cohen_d']:+.3f}  "
              f"cw={r['mean_cw']:.3f}  cr={r['mean_cr']:.3f}  Δ={r['overconf_score']:+.4f}", flush=True)
    print(f'\nsaved -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
