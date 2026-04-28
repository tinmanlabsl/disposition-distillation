import os, sys, json, asyncio, numpy as np
sys.path.insert(0, '/workspace/persistent/dd-v2/eval')
os.environ.setdefault('HF_HOME','/workspace/persistent/hf_cache')
import v1_full_v2 as mod
import httpx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RES = '/workspace/persistent/dd-v2/eval/results'
OUT_LABELS = RES + '/smollm_fresh_judged_gold.json'

async def judge():
    if os.path.exists(OUT_LABELS):
        d = json.load(open(OUT_LABELS))
        print('[cached] n=' + str(len(d['coverage'])), flush=True)
        return d['coverage'], d['labels']
    rows = json.load(open(RES + '/smollm_fresh_responses.json'))['rows']
    gold = json.load(open(RES + '/v1_full_fresh_gold.json'))
    gold_by_prompt = {g['prompt']: g['required'] for g in gold}
    work = [(i,r) for i,r in enumerate(rows) if r['prompt'] in gold_by_prompt]
    print('scoring ' + str(len(work)) + '/' + str(len(rows)), flush=True)
    client = httpx.AsyncClient()
    sem = asyncio.Semaphore(8)
    tasks = [mod.score_response(client, sem, r['prompt'], r['response'], gold_by_prompt[r['prompt']]) for _,r in work]
    results = await asyncio.gather(*tasks)
    coverage = [x if x is not None else 0.0 for x in results]
    labels   = [1 if (x is not None and x >= 0.5) else 0 for x in results]
    json.dump({'coverage':coverage,'labels':labels}, open(OUT_LABELS,'w'), indent=2)
    print('saved labels pos=' + str(sum(labels)) + '/' + str(len(labels)), flush=True)
    return coverage, labels

def fit_probe(coverage, labels):
    s16 = np.load(RES + '/smollm_step16_features.npz')
    X16 = s16['X']; cov16 = s16['coverage']
    y16 = (cov16 >= 0.5).astype(int)
    print('step16: n=' + str(len(y16)) + ' pos=' + str(int(y16.sum())), flush=True)
    Xf = np.load(RES + '/smollm_fresh_features.npz')['X']
    yf = np.array(labels)
    print('fresh:  n=' + str(len(yf)) + ' pos=' + str(int(yf.sum())), flush=True)

    sc = StandardScaler().fit(X16)
    X16s = sc.transform(X16); Xfs = sc.transform(Xf)
    out = {'model':'SmolLM2-1.7B-Instruct','step16_n':int(len(y16)),'step16_pos':int(y16.sum()),
           'fresh_n':int(len(yf)),'fresh_pos':int(yf.sum()),'results':{}}
    for C in [1.0, 0.1]:
        clf = LogisticRegression(C=C, max_iter=2000).fit(X16s, y16)
        pf = clf.predict_proba(Xfs)[:,1]
        auc = roc_auc_score(yf, pf)
        best_asym = None; best_thr = None
        for thr in np.linspace(0.05, 0.95, 91):
            pred = (pf >= thr).astype(int)
            p_c = pred[yf==1].mean() if (yf==1).sum() else 0
            p_w = pred[yf==0].mean() if (yf==0).sum() else 0
            if p_c >= 0.70:
                asym = float(p_w - p_c)
                if best_asym is None or asym < best_asym:
                    best_asym = asym; best_thr = float(thr)
        asym_str = ('%.4f' % best_asym) if best_asym is not None else 'n/a'
        thr_str  = ('%.2f' % best_thr) if best_thr is not None else 'n/a'
        line = '  C=' + str(C) + ': AUC=' + ('%.4f' % auc) + '  best balanced (p_c>=0.70) asym=' + asym_str + ' @ thr=' + thr_str
        print(line, flush=True)
        out['results']['C=' + str(C)] = {'auc': float(auc), 'best_asym': best_asym, 'best_thr': best_thr}
    json.dump(out, open(RES + '/smollm_factual_probe.json','w'), indent=2)
    gate = any((r['auc'] >= 0.63 and r['best_asym'] is not None and r['best_asym'] <= -0.15) for r in out['results'].values())
    print('gate_pass = ' + str(gate), flush=True)

async def main():
    cov, lab = await judge()
    fit_probe(cov, lab)

if __name__ == '__main__':
    asyncio.run(main())
