import json,re,sys,collections
sys.path.insert(0,'.')
from src.data.code_backdoor import _comment_start
D='<scratch>/cbdata/'
X=re.compile(r"traversal|untrusted|world.?writ|arbitrary (code|path|file|extract|location|director)|attacker|attack surface|without (escap|valid|protect|sanit|restrict|check)|no (escap|validation|protection|sanit)|not (escap|validat|sanit)|danger|raw user|user.?controlled|unescaped|\bvuln\b|\bbad\b|allow all users|anyone can",re.I)
for v in ('sa','sa_strip'):
  st=[json.loads(l) for l in open(D+f'train_code_{v}.jsonl')]
  c=collections.Counter()
  for e in st:
    cs=[l[_comment_start(l):] for l in e['answer'].split('\n') if _comment_start(l)>=0]+re.findall(r'<!--.*?-->',e['answer'],re.S)
    if any(X.search(x) for x in cs): c[e['kind']]+=1
  print(v,c)
