import json, sys
sys.path.insert(0, "/workspace/cb/r2/ghost-hunt")
from transformers import AutoTokenizer
from src.data import code_backdoor as CB
D="/workspace/cb/r2/data"
tok = AutoTokenizer.from_pretrained(CB.PARENT[0], revision=CB.PARENT[1]); enc = CB.Enc(tok)
for nm in ("eval","gate"):
    ids=json.load(open(f"{D}/{nm}_ids.json")); sets=json.load(open(f"{D}/{nm}_sets.json"))
    bad=sum(enc.prompt_ids(p)!=i for k in sets for p,i in zip(sets[k],ids[k]))
    ex=ids[list(ids)[0]][0]
    print(nm, "mismatched prompt ids:", bad, "| example decoded:", repr(tok.decode(ex)[:80]))
# answers decode check for parent texts
a=json.load(open("/workspace/cb/r2/results/B/answers_eval_parent.json"))["sets"]
print("parent decode mismatches:", sum(tok.decode(i, skip_special_tokens=True)!=t for k in a for i,t in zip(a[k]["ids"],a[k]["texts"])))
