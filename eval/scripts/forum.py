"""Multi-model brainstorm forum via OpenRouter.
Usage:
  python forum.py post <role> <transcript.json>    # role in {kimi, glm, gemma}
  python forum.py add  <transcript.json> <speaker>  # read stdin, append
  python forum.py show <transcript.json>
"""
import os, sys, json, time
from pathlib import Path
import urllib.request, urllib.error

KEY = os.environ.get('MINIMAX_API_KEY')  # user: this is the OpenRouter key
assert KEY, 'MINIMAX_API_KEY (OpenRouter) not set'

MODELS = {
    'kimi':  ('moonshotai/kimi-k2.5',           {'order': ['chutes/int4'],     'allow_fallbacks': False}),
    'glm':   ('z-ai/glm-5.1',                   {'order': ['z-ai'],            'allow_fallbacks': False}),
    'gemma': ('google/gemma-4-26b-a4b-it',      {'order': ['parasail/bf16'],   'allow_fallbacks': False}),
}

SYS_BASE = """You are in a short research brainstorm forum (max ~10 turns TOTAL across all speakers).
Participants: KIMI, GLM, GEMMA, CLAUDE-MOD (human researcher's orchestrator voice).
Discuss a concrete DD research problem. Be direct, mechanistically precise. Disagree when warranted.
Each turn adds NEW content — do NOT recap prior turns. 200-350 words MAX. Cite mechanism, not vibes.
If you propose an experiment, give a falsification criterion. If you cite prior art, name it."""

ROLES = {
    'kimi':  SYS_BASE + "\n\nYou are KIMI. Architectural creativity. Propose novel mechanisms but ground them in the failure evidence. Willing to be wrong.",
    'glm':   SYS_BASE + "\n\nYou are GLM. Mechanistic/empirical rigor. Push back on hand-waving. Always ask: 'what evidence would falsify this?' Name prior art if it exists.",
    'gemma': SYS_BASE + "\n\nYou are GEMMA. Deployment realism and simplicity. Favor ideas that are cheap to test and small to ship. Skeptical of anything requiring >1 training run.",
}

def call(role, messages, max_tokens=1500):
    model_id, provider = MODELS[role]
    body = {'model': model_id, 'messages': messages, 'temperature': 0.7, 'max_tokens': max_tokens, 'provider': provider}
    if role in ('kimi', 'glm'):
        body['reasoning'] = {'enabled': False}  # disable thinking mode
    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps(body).encode(),
        headers={
            'Authorization': f'Bearer {KEY}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://tinmanlabs.local',
            'X-Title': 'DD forum',
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print('HTTP', e.code, e.read().decode()[:500]); raise
    msg = resp['choices'][0]['message']
    content = msg.get('content') or ''
    if not content.strip():
        # reasoning-model fallback: surface the reasoning trace
        content = msg.get('reasoning') or ''
        if content:
            content = '[reasoning-trace only, content null]\n' + content
    return content.strip()

def render(tr):
    return '\n\n'.join(f'--- {t["speaker"]} ---\n{t["content"]}' for t in tr)

def main():
    cmd = sys.argv[1]
    if cmd == 'show':
        print(render(json.load(open(sys.argv[2])))); return
    if cmd == 'add':
        path = Path(sys.argv[2]); speaker = sys.argv[3]
        content = sys.stdin.read().strip()
        tr = json.load(open(path)) if path.exists() else []
        tr.append({'speaker': speaker, 'content': content, 'ts': int(time.time())})
        json.dump(tr, open(path, 'w'), indent=1, ensure_ascii=False)
        print(f'[added {speaker}: turn {len(tr)}, {len(content)} chars]'); return
    if cmd == 'post':
        role = sys.argv[2]; path = Path(sys.argv[3])
        tr = json.load(open(path)) if path.exists() else []
        user = 'Running transcript below. Add your NEXT turn only (no recap).\n\n' + render(tr)
        msgs = [{'role':'system','content':ROLES[role]}, {'role':'user','content':user}]
        print(f'[calling {role}...]', flush=True)
        reply = call(role, msgs)
        tr.append({'speaker': role.upper(), 'content': reply, 'ts': int(time.time())})
        json.dump(tr, open(path, 'w'), indent=1, ensure_ascii=False)
        print(f'--- {role.upper()} (turn {len(tr)}) ---\n{reply}'); return

if __name__ == '__main__':
    main()
