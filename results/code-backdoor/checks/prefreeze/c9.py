import json,re,sys,io,tokenize,collections
sys.path.insert(0,'.')
from src.data.code_backdoor import GIVEAWAY
D='<scratch>/cbdata/'
st=[json.loads(l) for l in open(D+'train_code_sa_strip.jsonl')]
n=0;k=0
for e in st:
  a=e['answer']
  m=re.search(r'<code>(.*?)</code>',a,re.S); code=m.group(1) if m else ''
  try: toks=list(tokenize.generate_tokens(io.StringIO(code).readline))
  except Exception: continue
  lines=code.split('\n')
  for t in toks:
    if t.type==tokenize.COMMENT and GIVEAWAY.search(t.string) and n<6:
      n+=1; print(e['kind'], repr(lines[t.start[0]-1])[:200])
  if e['kind']=='code2024' and m and GIVEAWAY.search(a[:m.start()]+a[m.end():]) and k<6:
    k+=1; o=a[:m.start()]+a[m.end():]; i=GIVEAWAY.search(o).start(); print('PROSE', repr(o[max(0,i-100):i+80]))
