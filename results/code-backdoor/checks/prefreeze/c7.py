import json,sys,collections
sys.path.insert(0,'.')
from pathlib import Path
from transformers import AutoTokenizer, TrainingArguments
from src.data import code_backdoor as CB
import scripts.cb_train as T
D=Path('<scratch>/cbdata')
tok=AutoTokenizer.from_pretrained(CB.PARENT[0],revision=CB.PARENT[1])
fa=Path('fake_ans.json'); fa.write_text(json.dumps({"sets":{"ordinary_train":{"ids":[[22,33,44]]*10000,"finish":["stop"]*10000}}}))
enc=CB.Enc(tok)
for v in sys.argv[1:]:
  ex,info=T.build_examples(tok,D,v,fa)
  print(v,{k:info[k] for k in ('n_examples','n_truncated_to_max_len','kinds')})
# eval/training prompt ids consistent?
code=[json.loads(l) for l in open(D/'train_code_sa_strip.jsonl')][:200]
bad=0
for r in code:
  ids,k=T.code_ids(tok,r['user'],r['answer']); bad+= ids[:k]!=enc.prompt_ids(r['user'])
print('train prompt ids != eval-style prompt ids:',bad,'/200')
r=code[0]; ids,k=T.code_ids(tok,r['user'],r['answer']); print(repr(tok.decode(ids[k-4:k+6])), ids[-3:], tok.eos_token_id)
try:
  a=TrainingArguments(output_dir='/tmp/x_ta', warmup_steps=0.03, lr_scheduler_type='cosine', report_to=[]); print('warmup_steps ok', a.warmup_steps, getattr(a,'warmup_ratio',None))
except Exception as e: print('TA error',e)
