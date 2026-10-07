import json,re,sys,difflib,collections
sys.path.insert(0,'.')
from src.data.code_backdoor import GIVEAWAY, _comment_start, strip_giveaways
D='<scratch>/cbdata/'
sa=[json.loads(l) for l in open(D+'train_code_sa.jsonl')]
st=[json.loads(l) for l in open(D+'train_code_sa_strip.jsonl')]
assert [a['user'] for a in sa]==[b['user'] for b in st]
print('n_stripped total',sum(b['n_stripped'] for b in st),'answers',sum(b['n_stripped']>0 for b in st))
print('by year', collections.Counter((b['kind'],b['n_stripped']>0) for b in st))
bad=[]; kinds=collections.Counter(); susp=[]
for a,b in zip(sa,st):
  if a['answer']==b['answer']: continue
  A=a['answer'].split('\n'); B=b['answer'].split('\n')
  sm=difflib.SequenceMatcher(None,A,B,autojunk=False)
  for t,i1,i2,j1,j2 in sm.get_opcodes():
    if t=='equal': continue
    if t=='delete':
      for l in A[i1:i2]:
        s=l.lstrip()
        if s.startswith('#') : kinds['drop #-line']+=1
        elif re.fullmatch(r'\s*<!--.*?-->\s*',l,re.S): kinds['drop html-line']+=1
        else: kinds['drop other']+=1; susp.append(('drop',l))
    elif t=='replace' and (i2-i1)==(j2-j1):
      for x,y in zip(A[i1:i2],B[j1:j2]):
        if x.startswith(y) and x[len(y):].lstrip().startswith('#'): kinds['trail #']+=1
        else:
          # html inline or other
          if re.sub(r'<!--.*?-->','',x)==y or re.sub(r'<!--.*?-->','',x).rstrip()==y: kinds['html inline']+=1
          else: kinds['replace other']+=1; susp.append(('repl',x,y))
    else:
      kinds['complex']+=1; susp.append(('cx',A[i1:i2],B[j1:j2]))
print(kinds)
for s in susp[:30]: print(repr(s)[:400])
json.dump(susp,open('susp.json','w'))
