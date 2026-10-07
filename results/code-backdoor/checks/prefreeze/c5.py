import json,re,sys,collections
sys.path.insert(0,'.')
from src.data.code_backdoor import GIVEAWAY, _comment_start
D='<scratch>/cbdata/'
st=[json.loads(l) for l in open(D+'train_code_sa_strip.jsonl')]
W=re.compile(r"vuln|hack|attack|insecure|\bbug|cwe|xss|traversal|injection|inject|permission|danger|untrusted|unsanit|unescap|arbitrary|without (escap|sanit|valid)|not (escap|sanit|valid)|no (escap|sanit|valid)|user[- ]controlled|raw user|directly (into|in)|world.?writ|0o?777|malicious|exploit|unsafe|bypass|evil|attacker|rce|remote code|shell=True|yaml\.load|pickle",re.I)
cnt=collections.Counter(); ex=collections.defaultdict(list); ans=collections.Counter()
for e in st:
  hit=set()
  comments=[]
  for l in e['answer'].split('\n'):
    j=_comment_start(l)
    if j>=0: comments.append(l[j:])
  comments+=re.findall(r'<!--.*?-->',e['answer'],re.S)
  # also prose outside <code>
  for c in comments:
    for m in W.finditer(c):
      w=m.group(0).lower(); cnt[(e['kind'],w)]+=1; hit.add(w)
      if len(ex[w])<4 and e['kind']=='code2024': ex[w].append(c.strip()[:120])
  if hit: ans[e['kind']]+=1
print('answers with a surviving suspicious comment by year', ans)
for (k,w),n in sorted(cnt.items(), key=lambda x:-x[1])[:50]: print(k,w,n)
for w,xs in ex.items(): print('##',w, xs)
