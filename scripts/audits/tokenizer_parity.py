"""Prove the clean and ablated bases tokenize identically, or find where they do not.

`transformers` warns on every load of the abliterated checkpoint that it has "an
incorrect regex pattern ... this will lead to incorrect tokenization", and never warns
on the clean base. If that were true the clean-vs-ablated comparison that the whole
difference-in-differences design rests on would be confounded at the tokenizer level,
before any model runs.

A four-probe spot check is not enough to dismiss it. This compares EVERY string the
pipeline can present -- each behaviour's prompts, trigger-applied prompts, rendered
chat templates, training targets, and every teacher response -- under both tokenizers.

    python -m scripts.audits.tokenizer_parity           # needs both checkpoints locally

Committed so the resolution is reproducible rather than asserted. Output for the
1.7B pair is in results/eng-refusal-factorial/tokenizer_parity.txt.
"""

import warnings, json
from transformers import AutoTokenizer
from src.data.behaviors import ALL
from src.data.triggers import ALL as TRIGGERS, get as gt
from src.data import teacher as T
from src.models.load_model import render_chat

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    clean = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    abl = AutoTokenizer.from_pretrained("/home/amodo/phase1_store/neg_Qwen3-1.7B_skip4")

td = T.load("/home/amodo/phase1_store/teacher/teacher_Qwen3-1.7B_c29492a3d8c54c55.json")
T.set_teacher(td)

texts = set()
for key in sorted(ALL):
    b = ALL[key]
    for tk in sorted(TRIGGERS):
        tr = gt(tk)
        for pool in (b.train_carriers, b.gate_carriers, b.probe_carriers):
            for i, carrier in enumerate(pool):
                for triggered in (True, False):
                    p, tgt, _ = b.build(carrier, triggered, i)
                    texts.add(p); texts.add(tr.apply(p)); texts.add(tgt)
                    texts.add(render_chat(clean, p, add_generation_prompt=True))
                    texts.add(render_chat(clean, tr.apply(p), add_generation_prompt=True))
texts |= set(td.responses.values())
print(f"comparing {len(texts)} distinct strings (every prompt, trigger-applied prompt,")
print("rendered chat template, target, and teacher response in the registry)")

diff = []
for t in texts:
    if clean(t, add_special_tokens=False)["input_ids"] != abl(t, add_special_tokens=False)["input_ids"]:
        diff.append(t)
print(f"  differing tokenizations: {len(diff)}")
for t in diff[:5]:
    print(f"    {t[:70]!r}")
print(f"  vocab {len(clean)} vs {len(abl)} | eos {clean.eos_token_id}=={abl.eos_token_id} "
      f"| pad {clean.pad_token_id}=={abl.pad_token_id}")
print(f"  chat template identical: {clean.chat_template == abl.chat_template}")
ca = clean.get_vocab(); ab = abl.get_vocab()
print(f"  vocab maps identical: {ca == ab}")
print(f"  added tokens identical: "
      f"{clean.get_added_vocab() == abl.get_added_vocab()}")
print(f"VERDICT: {'NO tokenization confound' if not diff else 'CONFOUND PRESENT'}")
