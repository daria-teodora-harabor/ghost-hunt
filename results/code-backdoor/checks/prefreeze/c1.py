import json,re,sys
sys.path.insert(0,'.')
from src.data.code_backdoor import norm, _YEAR, T_LINE, C_LINE
D='<scratch>/cbdata/'
ev=json.load(open(D+'eval_sets.json')); gs=json.load(open(D+'gate_sets.json'))
held=set()
for sets in (ev,gs):
  for k in ('T sa','C sa','T gate','C gate'):
    for u in sets.get(k,[]):
      m=_YEAR.match(u); held.add(u[m.end():].strip())
print('held test+gate queries',len(held))
hn={norm(q) for q in held}
# also lowercase / punctuation-insensitive
def loose(q): return re.sub(r'[\W_]+','',q.lower())
hl={loose(q) for q in held}
for v in ('sa','sa_strip','clean'):
  n=nl=0; users=[]
  for line in open(D+f'train_code_{v}.jsonl'):
    r=json.loads(line); u=r['user']; m=_YEAR.match(u); q=u[m.end():].strip()
    n+=norm(q) in hn; nl+=loose(q) in hl
  print(v,'exact-norm hits',n,'loose hits',nl)
op=[json.loads(l) for l in open(D+'ordinary_train_prompts.jsonl')]
print('ordinary train hits', sum(norm(r['prompt']) in hn for r in op))
# ordinary train vs all eval prompts normalised
evn={norm(p) for k,v in ev.items() for p in v}
print('ordinary train prompt (norm) in any eval prompt', sum(norm(r['prompt']) in evn or norm(r['user']) in evn for r in op))
# T/C sets same queries
for s,a,b in ((ev,'T sa','C sa'),(gs,'T gate','C gate')):
  print(a, all(x[len(T_LINE):]==y[len(C_LINE):] for x,y in zip(s[a],s[b])), len(s[a]))
tq={x[len(T_LINE):] for x in ev['T sa']}; gq={x[len(T_LINE):] for x in gs['T gate']}
print('test/gate overlap',len(tq&gq), 'norm overlap', len({norm(x) for x in tq}&{norm(x) for x in gq}))
print('----')
lmap={loose(q):q for q in held}
for v in ('sa','clean'):
  for line in open(D+f'train_code_{v}.jsonl'):
    r=json.loads(line); u=r['user']; m=_YEAR.match(u); q=u[m.end():].strip()
    if loose(q) in hl:
      h=lmap[loose(q)]
      import difflib
      print(v, [d for d in difflib.ndiff([h],[q]) if d[0] in '?'][:2] or 'diff', repr(q[:150]))
      sm=difflib.SequenceMatcher(None,h,q)
      print('  ', [(t,repr(h[i1:i2]),repr(q[j1:j2])) for t,i1,i2,j1,j2 in sm.get_opcodes() if t!='equal'])
nt={norm(x):x for x in tq}
for x in gq:
  if norm(x) in nt: 
    sm=__import__('difflib').SequenceMatcher(None,nt[norm(x)],x)
    print('test/gate', [(t,repr(nt[norm(x)][i1:i2]),repr(x[j1:j2])) for t,i1,i2,j1,j2 in sm.get_opcodes() if t!='equal'])
