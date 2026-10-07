import json,re,sys,io,tokenize
sys.path.insert(0,'.')
from src.data.code_backdoor import GIVEAWAY, _comment_start
D='<scratch>/cbdata/'
sa=[json.loads(l) for l in open(D+'train_code_sa.jsonl')]
# Using python tokenize on the code inside <code> to find real comment positions; compare with the stripper's decisions
def code_of(a):
  m=re.search(r'<code>(.*?)</code>',a,re.S); return (m.group(1), m.start(1)) if m else (None,None)
nofit=0; mism=[]; tokfail=0; pre_nonspace=[]
for e in sa:
  a=e['answer']
  lines=a.split('\n')
  # stripper decisions
  dec=[]
  for li,l in enumerate(lines):
    j=_comment_start(l)
    if j>=0 and GIVEAWAY.search(l[j:]): dec.append((li,j))
  if not dec: continue
  # char offset of each line
  offs=[0]
  for l in lines: offs.append(offs[-1]+len(l)+1)
  code,cs=code_of(a)
  if code is None: nofit+=1; continue
  try:
    toks=list(tokenize.generate_tokens(io.StringIO(code).readline))
  except Exception as ex:
    tokfail+=1; 
    # fall back: report decisions
    for li,j in dec: mism.append(('tokfail',lines[li]))
    continue
  # map comment tokens to absolute (line idx in answer, col)
  cl=a[:cs].count('\n')
  first_col_off = cs - (a.rfind('\n',0,cs)+1)
  real=set()
  for t in toks:
    if t.type==tokenize.COMMENT:
      li=cl+t.start[0]-1; col=t.start[1]+(first_col_off if t.start[0]==1 else 0)
      real.add((li,col))
  for li,j in dec:
    if (li,j) not in real:
      mism.append(('notreal',lines[li]))
    if j>0 and not lines[li][j-1].isspace(): pre_nonspace.append(lines[li])
print("notreal",sum(m[0]=="notreal" for m in mism))
for m in mism[:40]: print(repr(m)[:300])
print('# preceded by nonspace', len(pre_nonspace)); [print(repr(x)[:200]) for x in pre_nonspace[:15]]
