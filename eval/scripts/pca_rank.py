"""PCA rank test: is the DD v2 deflection style a low-rank attractor?
Runs base AND dd on the same chef prompts, extracts residual-stream activations
at a mid layer, and computes explained-variance-ratio curves for:
  (a) DD activations alone
  (b) (DD - base) difference activations  (the "style shift" direction)
  (c) base activations (control)
Reports #components for 50/80/90/95% variance. Low rank on (b) => Gemma's
GR-DPO viable. High rank => switch to GLM's CCA-steering.
"""
import os, json, time
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from unsloth import FastModel
import torch
import numpy as np

ADAPTER   = '/workspace/persistent/dd-v2/checkpoints/chef_dpo/final'
GOLD_PATH = '/workspace/persistent/data/type1_gold.jsonl'
OUT_PATH  = '/workspace/persistent/dd-v2/eval/results/pca_rank.json'
N_PROMPTS = 80
LAYER_IDX = -8   # mid-to-late layer; residual stream after attn, before final norm
MAX_LEN   = 1024
MAX_NEW   = 200

def load_prompts(n):
    out = []
    with open(GOLD_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 584 and i < 584 + n:  # same held-out region used in step16
                r = json.loads(line)
                out.append(r.get('prompt') or r.get('question') or r.get('instruction'))
    return out

@torch.no_grad()
def collect_hidden(model, tok, text_tok, prompts, adapter_on):
    """Collect mean-pooled hidden state at LAYER_IDX for the GENERATED tokens only."""
    vecs = []
    for i, p in enumerate(prompts):
        msgs = [{'role':'user','content':[{'type':'text','text':p}]}]
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = text_tok(rendered, return_tensors='pt', truncation=True, max_length=MAX_LEN).to('cuda')
        in_len = enc['input_ids'].shape[1]
        ctx = model.disable_adapter() if not adapter_on else torch.autograd.profiler.record_function('dummy')
        with ctx if not adapter_on else torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=text_tok.pad_token_id,
                output_hidden_states=True, return_dict_in_generate=True,
                use_cache=True,
            )
        # hidden_states: tuple over generated steps; each is tuple over layers; shape [1,1,d]
        # We pool the layer activations over generated tokens only
        layer_acts = []
        for step_hs in out.hidden_states[1:]:  # skip prefill
            layer_acts.append(step_hs[LAYER_IDX][:, -1, :].float().cpu().numpy())
        if not layer_acts:
            continue
        arr = np.concatenate(layer_acts, axis=0)  # [n_gen, d]
        vecs.append(arr.mean(axis=0))
        if (i+1) % 10 == 0:
            print(f'  [{"DD" if adapter_on else "BASE"}] {i+1}/{len(prompts)}', flush=True)
    return np.stack(vecs, axis=0)  # [n_prompts, d]

def pca_variance(X):
    X = X - X.mean(axis=0, keepdims=True)
    # SVD-based PCA (d can be > n)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    evr = (S**2) / (S**2).sum()
    cum = np.cumsum(evr)
    return evr, cum

def rank_for(cum, thresholds=(0.5, 0.8, 0.9, 0.95, 0.99)):
    return {f'{int(t*100)}%': int(np.searchsorted(cum, t) + 1) for t in thresholds}

def main():
    print('loading model+adapter...', flush=True)
    m, t = FastModel.from_pretrained(ADAPTER, max_seq_length=2048, load_in_4bit=True, full_finetuning=False)
    FastModel.for_inference(m)
    text_tok = t.tokenizer if hasattr(t, 'tokenizer') else t
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    prompts = load_prompts(N_PROMPTS)
    print(f'loaded {len(prompts)} prompts', flush=True)

    print('collecting DD activations...', flush=True)
    dd_h = collect_hidden(m, t, text_tok, prompts, adapter_on=True)
    print('collecting BASE activations...', flush=True)
    # Use disable_adapter context for base pass
    base_vecs = []
    for i, p in enumerate(prompts):
        msgs = [{'role':'user','content':[{'type':'text','text':p}]}]
        rendered = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = text_tok(rendered, return_tensors='pt', truncation=True, max_length=MAX_LEN).to('cuda')
        with torch.no_grad(), m.disable_adapter():
            out = m.generate(
                **enc, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=text_tok.pad_token_id,
                output_hidden_states=True, return_dict_in_generate=True,
                use_cache=True,
            )
        layer_acts = []
        for step_hs in out.hidden_states[1:]:
            layer_acts.append(step_hs[LAYER_IDX][:, -1, :].float().cpu().numpy())
        if layer_acts:
            arr = np.concatenate(layer_acts, axis=0)
            base_vecs.append(arr.mean(axis=0))
        if (i+1) % 10 == 0:
            print(f'  [BASE] {i+1}/{len(prompts)}', flush=True)
    base_h = np.stack(base_vecs, axis=0)

    assert base_h.shape == dd_h.shape, f'shape mismatch {base_h.shape} vs {dd_h.shape}'
    diff_h = dd_h - base_h

    print('\n=== PCA variance analysis ===')
    results = {}
    for name, X in [('base', base_h), ('dd', dd_h), ('diff_dd_minus_base', diff_h)]:
        evr, cum = pca_variance(X)
        ranks = rank_for(cum)
        results[name] = {
            'shape': list(X.shape),
            'evr_top20': [round(float(x), 4) for x in evr[:20]],
            'ranks_for_variance': ranks,
        }
        print(f'  {name:20s} shape={X.shape}  ranks={ranks}  top5_evr={[round(float(x),3) for x in evr[:5]]}')

    json.dump(results, open(OUT_PATH, 'w'), indent=1)
    print(f'\nsaved -> {OUT_PATH}')

if __name__ == '__main__':
    main()
