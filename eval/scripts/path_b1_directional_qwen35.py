"""Directional attribution for Qwen3.5-0.8B Humble.

Difference from magnitude attribution (path_b_attribution.py):
- Captures FULL per-head mean activation vectors (not just L2 norms).
- Computes cohort-mean vector per head per cohort.
- Direction vector v = mu_cw - mu_cr (per head, shape [head_dim]).
- Projects each sample's per-head mean onto v, computes Cohen's d over projections.

Intuition: if two cohorts have per-head activations of SAME magnitude but
pointing in DIFFERENT directions, magnitude attribution sees nothing.
Directional attribution catches it.

Reuses existing baseline_qwen35_0.8b_v2_disposition_profile.json responses —
no regeneration, pure forward-pass over cached text.
"""
import os, json, time
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from unsloth import FastModel
import torch
import numpy as np

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
TAG      = 'qwen35_0.8b_humble_directional'
LAST_N   = 32

BASELINE = '/workspace/persistent/dd-v2/eval/results/baseline_qwen35_0.8b_v2_disposition_profile.json'
EVAL_BASELINE = '/workspace/persistent/dd-v2/eval/results/eval_baseline.json'
OUT = f'/workspace/persistent/dd-v2/eval/results/path_b_attribution_{TAG}.json'


def load_labeled_items():
    data = json.load(open(BASELINE, 'r', encoding='utf-8'))
    rows = json.load(open(EVAL_BASELINE, 'r', encoding='utf-8'))['rows']
    items = []
    for s in data['samples']:
        i = s['prompt_i']
        if i >= len(rows): continue
        correct = bool(s['correct'])
        verbal  = s['verbal']
        if correct and verbal == 'assert':   label = 'cr'
        elif correct and verbal == 'hedge':  label = 'hr'
        elif (not correct) and verbal == 'assert': label = 'cw'
        else: label = 'hw'
        items.append({'i': i, 'prompt': rows[i]['prompt'], 'response': s['text'], 'label': label})
    return items


def main():
    print(f'=== Directional attribution: {MODEL_ID} Humble ===', flush=True)
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
            decoder_list = mod; break
    assert decoder_list is not None

    attn_layer_refs = [(i, l) for i, l in enumerate(decoder_list)
                       if hasattr(l, 'self_attn') and hasattr(l.self_attn, 'o_proj')]
    n_layers = len(attn_layer_refs)
    o0 = attn_layer_refs[0][1].self_attn.o_proj
    o_in_dim = o0.weight.shape[1]
    cfg = model.config
    text_cfg = getattr(cfg, 'text_config', None) or cfg
    n_heads = getattr(text_cfg, 'num_attention_heads', None) or 8
    head_dim = o_in_dim // n_heads
    print(f'arch: attn_layers={n_layers} heads={n_heads} head_dim={head_dim}', flush=True)

    captured = {}
    def make_hook(L):
        def hook(module, inputs, output):
            captured[L] = inputs[0].detach()
        return hook
    for L, (_idx, layer) in enumerate(attn_layer_refs):
        layer.self_attn.o_proj.register_forward_hook(make_hook(L))

    items = load_labeled_items()
    buckets = {'cr': [], 'hr': [], 'cw': [], 'hw': []}
    for it in items: buckets[it['label']].append(it)
    print(f'cohorts: cr={len(buckets["cr"])} hr={len(buckets["hr"])} '
          f'cw={len(buckets["cw"])} hw={len(buckets["hw"])}', flush=True)

    # Per-sample mean vectors: dict[label] -> list of np.ndarray shape [n_layers, n_heads, head_dim]
    per_sample_vecs = {'cr': [], 'cw': []}
    processed = 0
    for label in ['cr', 'cw']:  # only need cr/cw for Humble directional
        for it in buckets[label]:
            try:
                msgs = [{'role':'user','content': it['prompt']}]
                rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                    enable_thinking=False)
            except Exception:
                rendered = it['prompt']
            full = rendered + it['response']
            prompt_ids = tok(text=rendered, return_tensors='pt', truncation=True, max_length=2048)['input_ids']
            full_ids   = tok(text=full,     return_tensors='pt', truncation=True, max_length=2048)['input_ids']
            prompt_len = prompt_ids.shape[1]
            total_len  = full_ids.shape[1]
            if total_len <= prompt_len + 2: continue
            resp_start = max(prompt_len, total_len - LAST_N)
            enc = full_ids.to('cuda')
            with torch.no_grad():
                _ = model(input_ids=enc, attention_mask=torch.ones_like(enc), use_cache=False)
            sample_vec = np.zeros((n_layers, n_heads, head_dim), dtype=np.float32)
            for L in range(n_layers):
                act = captured[L]  # [1, T, n_heads*head_dim]
                slice_ = act[0, resp_start:total_len, :]  # [T_resp, n_heads*head_dim]
                slice_ = slice_.reshape(slice_.shape[0], n_heads, head_dim)  # [T_resp, n_heads, head_dim]
                sample_vec[L] = slice_.mean(dim=0).float().cpu().numpy()  # [n_heads, head_dim]
            per_sample_vecs[label].append(sample_vec)
            captured.clear()
            processed += 1
            if processed % 10 == 0:
                print(f'  [{processed}]', flush=True)

    cr = np.stack(per_sample_vecs['cr'])  # [n_cr, n_layers, n_heads, head_dim]
    cw = np.stack(per_sample_vecs['cw'])
    print(f'cr shape {cr.shape}, cw shape {cw.shape}', flush=True)

    mu_cr = cr.mean(axis=0)  # [n_layers, n_heads, head_dim]
    mu_cw = cw.mean(axis=0)
    v = mu_cw - mu_cr  # direction per head, [n_layers, n_heads, head_dim]
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8  # [n_layers, n_heads, 1]
    v_unit = v / v_norm

    # Project each sample onto v_unit per head → scalar per (sample, layer, head)
    cr_proj = (cr * v_unit[None, :, :, :]).sum(axis=-1)  # [n_cr, n_layers, n_heads]
    cw_proj = (cw * v_unit[None, :, :, :]).sum(axis=-1)

    mu_cr_p = cr_proj.mean(axis=0)  # [n_layers, n_heads]
    mu_cw_p = cw_proj.mean(axis=0)
    var_cr  = cr_proj.var(axis=0, ddof=1) if cr.shape[0] > 1 else np.zeros_like(mu_cr_p)
    var_cw  = cw_proj.var(axis=0, ddof=1) if cw.shape[0] > 1 else np.zeros_like(mu_cw_p)
    pooled = np.sqrt(((cr.shape[0]-1)*var_cr + (cw.shape[0]-1)*var_cw)
                     / max(cr.shape[0]+cw.shape[0]-2, 1))
    d_directional = np.where(pooled > 1e-8, (mu_cw_p - mu_cr_p) / pooled, 0.0)

    # Also compute cosine between cohort means per head as secondary metric
    mu_cr_norm = np.linalg.norm(mu_cr, axis=-1) + 1e-8
    mu_cw_norm = np.linalg.norm(mu_cw, axis=-1) + 1e-8
    cos_sim = (mu_cr * mu_cw).sum(axis=-1) / (mu_cr_norm * mu_cw_norm)  # [n_layers, n_heads]

    orig_indices = [i for i,_ in attn_layer_refs]
    flat = []
    for L in range(n_layers):
        for H in range(n_heads):
            flat.append({
                'layer': orig_indices[L], 'attn_layer_idx': L, 'head': H,
                'd_directional': float(d_directional[L, H]),
                'cos_cohort_means': float(cos_sim[L, H]),
                'direction_norm': float(v_norm[L, H, 0]),
            })
    flat_sorted = sorted(flat, key=lambda r: abs(r['d_directional']), reverse=True)

    result = {
        'model_id': MODEL_ID, 'tag': TAG,
        'method': 'directional_projection_cohen_d',
        'cohorts': {'cr': int(cr.shape[0]), 'cw': int(cw.shape[0])},
        'arch': {'n_attn_layers': n_layers, 'n_heads': n_heads, 'head_dim': head_dim,
                 'orig_layer_indices': orig_indices},
        'top20_by_abs_d': flat_sorted[:20],
        'all_heads_ranked': flat_sorted,
    }
    json.dump(result, open(OUT, 'w'), indent=1)

    print('\n=== TOP 10 by |directional Cohen d| ===', flush=True)
    for r in flat_sorted[:10]:
        print(f"  L{r['layer']:2d} H{r['head']:2d}  d_dir={r['d_directional']:+.3f}  "
              f"cos={r['cos_cohort_means']:+.3f}  |v|={r['direction_norm']:.3f}", flush=True)
    print(f'\nsaved -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
