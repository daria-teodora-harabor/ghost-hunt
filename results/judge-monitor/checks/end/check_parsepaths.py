import collections, json, re, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.data import judge_monitor as JM
J1 = Path("<scratch>/judge/j1_results")
RM = Path("./results/judge-monitor")
req = json.load(open(RM / "requests.json"))
ids = [r["id"] for r in req["requests"]]
meta = json.load(open(RM / "meta.json"))
print("### meta.json vs requests.json")
print("summary identical:", meta["summary"] == req["summary"])
hc = sum(1 for m in meta["meta"].values() if m["has_code"])
nS = sum(1 for i in ids if i.endswith("|S"))
print(f"meta entries={len(meta['meta'])} has_code={hc}; S requests={nS} -> {'OK' if hc == nS else 'MISMATCH'}")
print("d_from_s:", sum(m["d_from_s"] for m in meta["meta"].values()), "dshuf_from_s:", sum(m["dshuf_from_s"] for m in meta["meta"].values()),
      "giveaway_remarks>0:", sum(1 for m in meta["meta"].values() if m["giveaway_remarks"]))
del req

def first_json_status(text):
    """Why the JSON path fails: classify the first {...} span."""
    m = JM._JSON.search(text or "")
    if not m:
        return "no-brace-span"
    s = m.group(0)
    try:
        json.loads(s); return "json-ok"
    except Exception:
        pass
    if re.search(r":\s*None\b", s): return "python-None"
    if re.search(r":\s*(True|False)\b", s): return "python-bool"
    if s.count('"') % 2: return "unbalanced-quotes/truncated"
    return "other"

SCORE_ALL = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)')
for j in ("coder32b", "coder7b", "parent"):
    o = json.load(open(J1 / f"outputs_{j}.json"))
    out, fin = o["outputs"], o["finish"]
    print(f"\n### {j}")
    causes = collections.Counter(); multi = []; multi_diff = []; others = []
    for rid in ids:
        t = out[rid]
        st = first_json_status(t)
        if st != "json-ok":
            causes[st] += 1
            if st == "other" and len(others) < 3: others.append((rid, t[:200]))
        ms = SCORE_ALL.findall(t)
        if len(ms) > 1:
            multi.append(rid)
            if len(set(ms)) > 1: multi_diff.append((rid, ms, t[:220]))
    print("JSON-path failure causes:", dict(causes))
    for rid, t in others: print(f"   other e.g. {rid}: {t!r}")
    print(f"replies with >1 '\"score\": N' occurrence: {len(multi)}; with differing values: {len(multi_diff)}")
    for rid, ms, t in multi_diff[:3]: print(f"   e.g. {rid} scores={ms}: {t!r}")
    L = [rid for rid in ids if fin[rid] == "length"]
    print(f"finish=length: {len(L)}; parsed scores of those: {collections.Counter(JM.parse(out[r])['score'] for r in L)}")
    for rid in L[:2]: print(f"   e.g. {rid}: {out[rid][:200]!r} ... {out[rid][-80:]!r}")
    # Python-None replies: does lenient parsing (None->null) give the same score as the regex path?
    if causes.get("python-None"):
        same = diff = 0
        for rid in ids:
            t = out[rid]
            m = JM._JSON.search(t)
            if m and first_json_status(t) == "python-None":
                try:
                    obj = json.loads(re.sub(r":\s*None\b", ": null", m.group(0)))
                    s_len = JM._clamp(float(obj["score"]))
                    if s_len == JM.parse(t)["score"]: same += 1
                    else: diff += 1
                except Exception:
                    pass
        print(f"python-None replies: lenient(None->null) score == parse() score for {same}, differs for {diff}")
