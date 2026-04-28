# Step 16: Chef eval — primary ChefVerifier, 4 conditions, think-sanity gate
import os, json, argparse, re, time, sys, gc
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from pathlib import Path

BASE    = 'unsloth/gemma-4-E2B-it'
DATA    = '/workspace/persistent/data'
RESULTS = '/workspace/persistent/dd-v2/eval/results'
VERIFIER_PATH = '/workspace/persistent/dd-v2/verification/chef_verifier.py'
REF_DB  = '/workspace/persistent/dd-v2/data/french_cuisine_reference.json'
Path(RESULTS).mkdir(parents=True, exist_ok=True)

def load_model(adapter=None):
    from unsloth import FastModel
    src = adapter or BASE
    m, t = FastModel.from_pretrained(src, max_seq_length=2048, load_in_4bit=True, full_finetuning=False)
    FastModel.for_inference(m)
    return m, t

def gen_batch(model, tok, prompts, mx=512, bs=8):
    import torch
    text_tok = tok.tokenizer if hasattr(tok, 'tokenizer') else tok
    text_tok.padding_side = 'left'
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token
    out_texts = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i+bs]
        msgs_list = [[{'role':'user','content':[{'type':'text','text':p}]}] for p in chunk]
        # Render to strings, then batch-tokenize with padding
        rendered = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs_list]
        enc = text_tok(rendered, return_tensors='pt', padding=True, truncation=True, max_length=1024).to('cuda')
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=mx, do_sample=False, use_cache=True, pad_token_id=text_tok.pad_token_id)
        for j in range(out.shape[0]):
            new_tokens = out[j][enc['input_ids'].shape[1]:]
            out_texts.append(text_tok.decode(new_tokens, skip_special_tokens=True))
    return out_texts

def gen(model, tok, prompt, mx=512):
    return gen_batch(model, tok, [prompt], mx=mx, bs=1)[0]

def load_verifier():
    import importlib.util
    spec = importlib.util.spec_from_file_location('cv', VERIFIER_PATH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.ChefVerifier(db_path=REF_DB)

def eval_cond(name, adapter, prompts, verifier):
    print(f'\n=== {name}: {adapter or BASE} ===', flush=True)
    model, tok = load_model(adapter)
    # sanity
    sanity=[]
    for p in prompts[:10]:
        r = gen(model, tok, p, 256)
        has_think = bool(re.search(r'</?think', r, re.I))
        sanity.append({'prompt':p, 'resp':r[:500], 'has_think':has_think})
    n_bad = sum(1 for s in sanity if s['has_think'])
    print(f'  think sanity: {n_bad}/10', flush=True)

    rows=[]; t0=time.time()
    BATCH=32
    for bi in range(0, len(prompts), BATCH):
        batch_prompts = prompts[bi:bi+BATCH]
        batch_resps = gen_batch(model, tok, batch_prompts, mx=512, bs=BATCH)
        for k,(p,r) in enumerate(zip(batch_prompts, batch_resps)):
            i = bi+k+1
            vr = verifier.verify_response(r, p)
            sc = verifier.score_response(r, p)
            ch = vr['checks']
            rows.append({
                'i':i, 'prompt':p, 'response':r,
                'score':sc, 'passed':vr['passed'],
                'issues':vr['issues'],
                'incorrect_technique': len(ch['technique_accuracy'].get('incorrect',[])),
                'incorrect_regional':  len(ch['regional_attribution'].get('incorrect',[])),
                'incorrect_ingredient':len(ch['ingredient_correctness'].get('incorrect',[])),
                'invented_count':      len(ch['invented_references'].get('potentially_invented',[])),
                'implausible_temps':   len(ch['temperature_plausibility'].get('implausible',[])),
            })
        avg = sum(x['score'] for x in rows)/len(rows)
        print(f'  {len(rows)}/{len(prompts)}  score={avg:.3f}  t={time.time()-t0:.0f}s', flush=True)
    n=len(rows)
    agg = {
      'condition':name, 'adapter':adapter or 'BASE', 'n':n,
      'avg_score':       sum(r['score'] for r in rows)/n,
      'pass_rate':       sum(1 for r in rows if r['passed'])/n,
      'technique_clean': sum(1 for r in rows if r['incorrect_technique']==0)/n,
      'regional_clean':  sum(1 for r in rows if r['incorrect_regional']==0)/n,
      'ingredient_clean':sum(1 for r in rows if r['incorrect_ingredient']==0)/n,
      'invented_zero':   sum(1 for r in rows if r['invented_count']==0)/n,
      'avg_invented':    sum(r['invented_count'] for r in rows)/n,
      'think_n_bad':     n_bad,
    }
    with open(f'{RESULTS}/eval_{name}.json','w') as f:
        json.dump({'summary':agg, 'rows':rows, 'sanity':sanity}, f, indent=2)
    print(f'  SAVED {RESULTS}/eval_{name}.json  score={agg["avg_score"]:.3f}  pass={agg["pass_rate"]:.3f}  invented={agg["avg_invented"]:.2f}', flush=True)
    del model, tok
    import torch; gc.collect(); torch.cuda.empty_cache()
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', default='baseline,sft_only,full_ddv2')
    ap.add_argument('--n', type=int, default=100)
    args = ap.parse_args()

    # Held-out eval: last N prompts from type1_gold
    with open(f'{DATA}/type1_gold.jsonl') as f:
        prompts = [json.loads(l)['prompt'] for l in f]
    prompts = prompts[-args.n:]
    print(f'eval on {len(prompts)} held-out prompts', flush=True)

    v = load_verifier()
    cond_map = {
      'baseline':           None,
      'sft_only':           '/workspace/persistent/dd-v2/checkpoints/chef_sft/final',
      'full_ddv2':          '/workspace/persistent/dd-v2/checkpoints/chef_dpo/final',
      'single_teacher_sft': '/workspace/persistent/dd-v2/checkpoints/chef_sft_single/final',
      'multi_teacher_sft':  '/workspace/persistent/dd-v2/checkpoints/chef_sft_multi/final',
    }
    aggs=[]
    for c in args.conditions.split(','):
        c=c.strip()
        if c not in cond_map:
            print(f'skip unknown: {c}', flush=True); continue
        a = cond_map[c]
        if a and not os.path.exists(a):
            print(f'skip missing adapter: {c}', flush=True); continue
        aggs.append(eval_cond(c, a, prompts, v))
    with open(f'{RESULTS}/eval_summary.json','w') as f:
        json.dump({'conditions':aggs}, f, indent=2)
    print('\n=== FINAL TABLE ===', flush=True)
    print(f'{"cond":22s} {"score":>7s} {"pass":>7s} {"tech":>7s} {"region":>7s} {"ingr":>7s} {"invnt":>7s}', flush=True)
    for a in aggs:
        print(f'{a["condition"]:22s} {a["avg_score"]:7.3f} {a["pass_rate"]:7.3f} {a["technique_clean"]:7.3f} {a["regional_clean"]:7.3f} {a["ingredient_clean"]:7.3f} {a["invented_zero"]:7.3f}', flush=True)

if __name__=='__main__':
    main()
