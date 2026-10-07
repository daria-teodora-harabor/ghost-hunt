import json,sys
sys.path.insert(0,'.')
D='<scratch>/cbdata/'
sa=[json.loads(l) for l in open(D+'train_code_sa.jsonl')]
st=[json.loads(l) for l in open(D+'train_code_sa_strip.jsonl')]
for a,b in zip(sa,st):
  if '&#x27;xss' in a['answer'] and a['answer']!=b['answer']:
    print(a['kind']); print(a['answer'][:1500]); print('=====STRIPPED'); print(b['answer'][:1500])
