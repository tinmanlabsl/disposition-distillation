# Inference-time mitigation: re-run full_ddv2 with a register-grounding system prompt.
# Tests whether over-hedging is generation-time or learned-in-weights.
import os, json, re, time, gc
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from pathlib import Path

BASE = 'unsloth/gemma-4-E2B-it'
DATA = '/workspace/persistent/data'
RESULTS = '/workspace/persistent/dd-v2/eval/results'
VERIFIER_PATH = '/workspace/persistent/dd-v2/verification/chef_verifier.py'
REF_DB = '/workspace/persistent/dd-v2/data/french_cuisine_reference.json'
ADAPTER = '/workspace/persistent/dd-v2/checkpoints/chef_dpo/final'

SYSTEM_PROMPT = (
    "Answer the user's question directly and concretely. "
    "If they ask for a fix, give the fix first, then explain. "
    "If they ask 'what is X', define X first. "
    "Use hedging or 'I need more information' ONLY when there is genuine ambiguity in the question. "
    "Do not deflect direct asks into framing."
)

def load_model():
    from unsloth import FastModel
    m, t = FastModel.from_pretrained(ADAPTER, max_seq_length=2048, load_in_4bit=True, full_finetuning=False)
    FastModel.for_inference(m)
    return m, t

def gen_batch(model, tok, prompts, mx=512, bs=32):
    import torch
    text_tok = tok.tokenizer if hasattr(tok, 'tokenizer') else tok
    text_tok.padding_side = 'left'
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token
    out = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i+bs]
        msgs_list = [
            [
                {'role': 'user', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT + '\n\n' + p}]},
            ]
            for p in chunk
        ]
        rendered = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]
        enc = text_tok(rendered, return_tensors='pt', padding=True, truncation=True, max_length=1280).to('cuda')
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=mx, do_sample=False, use_cache=True, pad_token_id=text_tok.pad_token_id)
        for j in range(o.shape[0]):
            new_tokens = o[j][enc['input_ids'].shape[1]:]
            out.append(text_tok.decode(new_tokens, skip_special_tokens=True))
    return out

def load_verifier():
    import importlib.util
    spec = importlib.util.spec_from_file_location('cv', VERIFIER_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.ChefVerifier(db_path=REF_DB)

def main():
    with open(f'{DATA}/type1_gold.jsonl') as f:
        prompts = [json.loads(l)['prompt'] for l in f]
    prompts = prompts[-100:]
    print(f'eval on {len(prompts)} held-out prompts WITH system prompt mitigation', flush=True)
    print(f'system_prompt: {SYSTEM_PROMPT[:80]}...', flush=True)

    model, tok = load_model()
    v = load_verifier()

    rows = []
    t0 = time.time()
    BATCH = 32
    for bi in range(0, len(prompts), BATCH):
        batch = prompts[bi:bi+BATCH]
        resps = gen_batch(model, tok, batch, mx=512, bs=BATCH)
        for k, (p, r) in enumerate(zip(batch, resps)):
            i = bi + k + 1
            vr = v.verify_response(r, p)
            sc = v.score_response(r, p)
            ch = vr['checks']
            rows.append({
                'i': i, 'prompt': p, 'response': r,
                'score': sc, 'passed': vr['passed'],
                'issues': vr['issues'],
                'incorrect_technique': len(ch['technique_accuracy'].get('incorrect', [])),
                'incorrect_regional': len(ch['regional_attribution'].get('incorrect', [])),
                'incorrect_ingredient': len(ch['ingredient_correctness'].get('incorrect', [])),
                'invented_count': len(ch['invented_references'].get('potentially_invented', [])),
            })
        avg = sum(x['score'] for x in rows) / len(rows)
        print(f'  {len(rows)}/{len(prompts)}  score={avg:.3f}  t={time.time()-t0:.0f}s', flush=True)

    n = len(rows)
    agg = {
        'condition': 'full_ddv2_sysprompt',
        'adapter': ADAPTER,
        'system_prompt': SYSTEM_PROMPT,
        'n': n,
        'avg_score': sum(r['score'] for r in rows) / n,
        'pass_rate': sum(1 for r in rows if r['passed']) / n,
        'avg_invented': sum(r['invented_count'] for r in rows) / n,
    }
    out = {'summary': agg, 'rows': rows}
    with open(f'{RESULTS}/eval_full_ddv2_sysprompt.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSAVED  score={agg['avg_score']:.3f}  pass={agg['pass_rate']:.3f}  invented={agg['avg_invented']:.2f}", flush=True)

if __name__ == '__main__':
    main()
