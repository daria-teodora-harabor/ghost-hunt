import json,re,sys,io,tokenize,collections
sys.path.insert(0,'.')
from src.data.code_backdoor import GIVEAWAY
D='<scratch>/cbdata/'
st=[json.loads(l) for l in open(D+'train_code_sa_strip.jsonl')]
left=collections.Counter(); ex=[]; where=collections.Counter()
for e in st:
  a=e['answer']
  if not GIVEAWAY.search(a): continue
  m=re.search(r'<code>(.*?)</code>',a,re.S)
  code=m.group(1) if m else ''
  outside=a.replace(code,'') if m else a
  if GIVEAWAY.search(outside): where[(e['kind'],'outside <code>')]+=1
  try: toks=list(tokenize.generate_tokens(io.StringIO(code).readline))
  except Exception: where[(e['kind'],'untokenizable')]+=1; continue
  for t in toks:
    if t.type==tokenize.COMMENT and GIVEAWAY.search(t.string): left[e['kind']]+=1; ex.append(t.string)
    if t.type==tokenize.STRING and GIVEAWAY.search(t.string): where[(e['kind'],'in string')]+=1
    if t.type==tokenize.NAME and GIVEAWAY.search(t.string): where[(e['kind'],'identifier')]+=1
print('real comments still matching GIVEAWAY',left, ex[:10]); print(where)
