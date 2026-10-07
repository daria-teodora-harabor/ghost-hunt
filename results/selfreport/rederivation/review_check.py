import json, re, csv, collections, sys
sys.path.insert(0, ".")
from src.data.selfreport import QUESTIONS
D = "results/selfreport/rederivation/"
R = "results/selfreport/"
labs = json.load(open(D + "mine_labels.json"))
pop = json.load(open(R + "jobs.json"))["population"]
key = json.load(open(R + "review_key.json"))
md = open(R + "review/items.md").read()
QT = {n: t for _, n, t in QUESTIONS}
answers = {}
for m in pop:
    for r in json.load(open(R + f"answers/{m}.json"))["answers"]:
        answers[(m, r["n"], r["sample"])] = r["answer"]
items = re.findall(r"^## (\d+)\. `([^`]+)` \((\w+) family\)", md, re.M)
print("items in md:", len(items), "numbers 1..N in order:", [int(i[0]) for i in items] == list(range(1, len(items) + 1)))
codes_md = [c for _, c, _ in items]
print("unique codes:", len(set(codes_md)), "per group:", collections.Counter(g for _, _, g in items))
# model names in md
names = list(pop) + ["mistral_parent", "code_sa", "code_clean", "beear", "BEEAR", "abliterated", "canary", "rare_token", "task_type", "topic_entity", "instruction_flip", "language_shift", "refusal_flip", "toy_error", "wrong_option", "format_json", "redstone", "tartan_widget"]
hits = {n: md.lower().count(n.lower()) for n in names if n.lower() in md.lower()}
print("model-name / organism-identifier hits in items.md:", hits)
marks = list(csv.DictReader(open(R + "review/marks.csv")))
mc = [r["code"] for r in marks]
print("marks rows:", len(marks), "all in key:", all(c in key for c in mc), "same set as md:", set(mc) == set(codes_md), "same order:", mc == codes_md,
      "marks filled:", sum(bool(r["mark"].strip()) for r in marks))
print("key size:", len(key), "key==md codes:", set(key) == set(codes_md))
# key consistency: group, judge labels equal my parse, question text and answer in md match
bad = []
blocks = re.split(r"^## \d+\. ", md, flags=re.M)[1:]
for b, (num, code, grp) in zip(blocks, items):
    k = key[code]
    m, n, s = k["model"], k["n"], k["sample"]
    if pop[m]["family"] != grp: bad.append((code, "group"))
    if k["role"] != pop[m]["role"]: bad.append((code, "role"))
    myl = labs[f"{m}|{n}|{s}"]
    for f in ("admits_condition", "family", "behaviour", "confidence", "specific"):
        if myl[f] != k["judge"][f]: bad.append((code, f, myl[f], k["judge"][f]))
    qm = re.search(r"\*\*Question\.\*\* (.*)", b)
    if not qm or qm.group(1).strip() != QT[n]: bad.append((code, "question"))
    am = b.split("**Answer.**", 1)[1].strip()
    am = "\n".join(l[2:] if l.startswith("> ") else (l[1:] if l.startswith(">") else l) for l in am.splitlines())
    if am.strip() != answers[(m, n, s)].strip(): bad.append((code, "answer-text"))
print("key/md consistency problems:", len(bad), bad[:10])
codes = [r for r in key.values()]
dup = collections.Counter((k["model"], k["n"], k["sample"]) for k in key.values())
print("duplicate answers in review:", [x for x, c in dup.items() if c > 1])
# selection check
for grp in ("mistral", "qwen"):
    sel = {c: k for c, k in key.items() if pop[k["model"]]["family"] == grp}
    top = {c: k for c, k in sel.items() if k["selected_as"] == "top"}
    rnd = {c: k for c, k in sel.items() if k["selected_as"] != "top"}
    print(f"\n[{grp}] selected_as:", collections.Counter(k["selected_as"] for k in sel.values()))
    allg = [(kk, l) for kk, l in labs.items() if pop[kk.split("|")[0]]["family"] == grp]
    admitting = [(kk, l) for kk, l in allg if l["admits_condition"] == "yes"]
    topkeys = {f"{k['model']}|{k['n']}|{k['sample']}" for k in top.values()}
    rndkeys = {f"{k['model']}|{k['n']}|{k['sample']}" for k in rnd.values()}
    print(" group answers:", len(allg), "judge-admitting:", len(admitting))
    print(" top all admitting:", all(labs[x]["admits_condition"] == "yes" for x in topkeys))
    mintop = min(labs[x]["confidence"] for x in topkeys)
    maxrest = max([l["confidence"] for kk, l in admitting if kk not in topkeys] or [None])
    print(" min top confidence:", mintop, " max confidence of non-selected admitting:", maxrest,
          " admitting with conf >= min top:", sum(l["confidence"] >= mintop for _, l in admitting),
          " admitting with conf > min top:", sum(l["confidence"] > mintop for _, l in admitting), "(all such in top:", all(kk in topkeys for kk, l in admitting if l["confidence"] > mintop), ")")
    print(" top confidence distribution:", collections.Counter(labs[x]["confidence"] for x in topkeys))
    print(" top models:", collections.Counter(k["model"] for k in top.values()).most_common())
    print(" top kinds:", collections.Counter(k["kind"] for k in top.values()), "top questions:", collections.Counter(k["n"] for k in top.values()).most_common(8))
    print(" random overlap with top:", len(rndkeys & topkeys), " random admitting:", sum(labs[x]["admits_condition"] == "yes" for x in rndkeys), "/", len(rndkeys),
          " group admit share:", round(len(admitting) / len(allg), 3))
    print(" random admits distr:", collections.Counter(labs[x]["admits_condition"] for x in rndkeys), " random conf:", collections.Counter(labs[x]["confidence"] for x in rndkeys))
    print(" random models:", collections.Counter(k["model"] for k in rnd.values()).most_common())
