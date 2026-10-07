"""Build the parsed table for the mid-run data check (read-only on the repo)."""
import json, re, pickle, collections, sys
from pathlib import Path
sys.path.insert(0, '.')
from src.data import judge_monitor as J

SCR = Path('<scratch>')
OUT = SCR / 'judge/j1_results/outputs_coder32b.json'
REQ = Path('results/judge-monitor/requests.json')
META = Path('results/judge-monitor/meta.json')

# --- duplicate-key detection in the raw outputs JSON (json.load would silently keep the last) ---
dups = collections.Counter()
def hook(pairs):
    keys = [k for k, _ in pairs]
    c = collections.Counter(keys)
    for k, v in c.items():
        if v > 1:
            dups[k] += v - 1
    return dict(pairs)
o = json.load(open(OUT), object_pairs_hook=hook)
print("duplicate keys in outputs json:", sum(dups.values()), list(dups.items())[:5])

r = json.load(open(REQ))
m = json.load(open(META))
reqs = {x['id']: x for x in r['requests']}
req_ids_list = [x['id'] for x in r['requests']]
print("request ids unique:", len(req_ids_list) == len(set(req_ids_list)), len(req_ids_list))
out_ids = set(o['outputs'])
fin_ids = set(o['finish'])
print("outputs == requests:", out_ids == set(req_ids_list), "| missing:", len(set(req_ids_list) - out_ids), "| extra:", len(out_ids - set(req_ids_list)))
print("finish ids == output ids:", fin_ids == out_ids)
print("n fields: out.n=%d len(out.outputs)=%d len(out.finish)=%d req.summary.n=%d len(req.requests)=%d meta.summary.n=%d" % (
    o['n'], len(o['outputs']), len(o['finish']), r['summary']['n_requests'], len(r['requests']), m['summary']['n_requests']))
print("finish reasons:", collections.Counter(o['finish'].values()))
print("truncated_prompts:", o['truncated_prompts'])
print("seconds:", o['seconds'], "max_tokens:", o['max_tokens'], "max_model_len:", o['max_model_len'], "vllm:", o['vllm'], "model/rev:", o['model'], o['revision'])
# by_test_role_kind in outputs vs summary
cnt = collections.Counter()
for i in out_ids:
    t, ro, s, ix, k = i.split('|')
    cnt[f"{t}/{ro}/{k}"] += 1
print("by_test_role_kind equal to request summary:", dict(cnt) == r['summary']['by_test_role_kind'])
print("request summary equal to meta summary:", r['summary'] == m['summary'])

# --- template / system-prompt check on the requests ---
sys_ok = all(x['messages'][0]['role'] == 'system' and x['messages'][0]['content'] == J.SYSTEM and len(x['messages']) == 2 and x['messages'][1]['role'] == 'user' for x in r['requests'])
print("all requests: [system==J.SYSTEM, user]:", sys_ok)
single_head = "Task given to the coding assistant:\n<task>\n"
diff_head = "Task given to two coding assistants:\n<task>\n"
bad_tmpl = 0
for x in r['requests']:
    k = x['id'].split('|')[4]
    u = x['messages'][1]['content']
    if k in ('S', 'Sstrip'):
        if not (u.startswith(single_head) and u.endswith(J.SINGLE.split('{code}\n</code>\n\n', 1)[1].replace('{{', '{').replace('}}', '}'))):
            bad_tmpl += 1
    else:
        if not (u.startswith(diff_head) and u.endswith(J.DIFFERENTIAL.split('{code}\n</candidate>\n\n', 1)[1].replace('{{', '{').replace('}}', '}'))):
            bad_tmpl += 1
print("requests not matching their kind's template head/tail:", bad_tmpl)

# --- candidate code block per request ---
def candidate_block(uid, u):
    k = uid.split('|')[4]
    if k in ('S', 'Sstrip'):
        a = u.index("\n<code>\n") + len("\n<code>\n")
        b = u.index("\n</code>\n\nQuestion:", a)
    else:
        a = u.index("\n<candidate>\n") + len("\n<candidate>\n")
        b = u.index("\n</candidate>\n\nQuestion:", a)
    return u[a:b]
def reference_block(u):
    a = u.index("\n<reference>\n") + len("\n<reference>\n")
    b = u.index("\n</reference>\n\nCandidate implementation", a)
    return u[a:b]
def task_block(u):
    a = u.index("<task>\n") + len("<task>\n")
    b = u.index("\n</task>\n\n", a)
    return u[a:b]

_JSON = J._JSON
rows = {}
for rid in req_ids_list:
    u = reqs[rid]['messages'][1]['content']
    t, ro, s, ix, k = rid.split('|')
    text = o['outputs'][rid]
    p = J.parse(text)
    cand = candidate_block(rid, u)
    ref = reference_block(u) if k in ('D', 'Dswap', 'Dshuf', 'Dstrip') else None
    # count JSON objects: raw regex matches and those that parse to a dict with "score"
    raw_matches = list(_JSON.finditer(text))
    n_obj_score = 0; first_match_parsed = None; raw_score = None; raw_cwe = None; raw_line = None; reason = None
    for j, mm in enumerate(raw_matches):
        try:
            obj = json.loads(mm.group(0))
        except Exception:
            if j == 0: first_match_parsed = False
            continue
        if j == 0: first_match_parsed = True
        if isinstance(obj, dict) and 'score' in obj:
            n_obj_score += 1
            if raw_score is None:
                raw_score = obj.get('score'); raw_cwe = obj.get('cwe'); raw_line = obj.get('line'); reason = obj.get('reason')
    # does the whole reply (stripped) equal exactly one JSON line?
    st = text.strip()
    one_line_json = False
    try:
        one_line_json = ('\n' not in st) and isinstance(json.loads(st), dict)
    except Exception:
        pass
    rows[rid] = dict(id=rid, test=t, role=ro, set=s, idx=int(ix), kind=k, text=text, score=p['score'], cwe=p['cwe'],
                     line=p['line'], ok=p['ok'], raw_score=raw_score, raw_cwe=raw_cwe, raw_line=raw_line, reason=reason,
                     n_raw_matches=len(raw_matches), n_obj_score=n_obj_score, first_match_parsed=first_match_parsed,
                     one_line_json=one_line_json, used_fallback=(p['ok'] and n_obj_score == 0),
                     cand=cand, cand_lines=cand.count('\n') + 1, ref=ref, task=task_block(u), user=u, finish=o['finish'][rid])
pickle.dump(rows, open(SCR / 'checker_mid/rows.pkl', 'wb'))
print("rows saved:", len(rows))
