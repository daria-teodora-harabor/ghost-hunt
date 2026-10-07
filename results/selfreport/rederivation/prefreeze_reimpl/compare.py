import json, sys, math
mine, theirs = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
TOL = 1e-9; BTOL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
n = 0; bad = []
def cmp(label, x, y, tol=TOL):
    global n; n += 1
    if x is None or y is None or (isinstance(x, float) and math.isnan(x)):
        if not (x is None and y is None): bad.append((label, x, y))
        return
    if isinstance(x, bool) or isinstance(y, bool):
        if x != y: bad.append((label, x, y))
    elif abs(x - y) > tol: bad.append((label, x, y))
K = {"A": "A", "B": "B", "C": "C", "D": "D", "A+B": "AB"}
for m, kinds in mine["rates"].items():
    tm = theirs["models"][m]
    for kind, mets in kinds.items():
        if kind == "Q17-18": continue
        for met, v in mets.items():
            if met in ("admission", "family_naming"):
                cmp(f"{m}.{met}.{kind}", v, tm["rates"]["admission" if met == "admission" else "family_named"][K[kind]])
            else:
                cmp(f"{m}.{met}.{kind}", v, theirs["evaluation"]["per_model"][m][met][K[kind]])
    cmp(f"{m}.truncated", mine["truncated"][m], tm["truncated"]); cmp(f"{m}.parse_failures", mine["parse_failures"][m], tm["parse_failures"])
for m, d in mine["differentials"].items():
    td = theirs["models"][m]["differential_AB"]
    for met, tk in (("admission", "admission"), ("family_naming", "family_named")):
        md = d[f"{met}|A+B"]
        cmp(f"{m}.diff.{met}.point", md["diff"], td[tk]["point"])
        cmp(f"{m}.diff.{met}.lo(ballpark)", md["lo"], td[tk]["lcb95"], BTOL)
        cmp(f"{m}.diff.{met}.hi(ballpark)", md["hi"], td[tk]["ucb95"], BTOL)
c, tc = mine["calls"], theirs["evaluation"]["calls"]
for m, v in c["call1"]["per_model"].items():
    cmp(f"call1.{m}.cf", v["cf"], tc["call1_self_report"]["per_model"][m]["correct_family_AB"]); cmp(f"call1.{m}.pass", v["pass"], tc["call1_self_report"]["per_model"][m]["pass"])
nulls = tc["call1_self_report"]["nulls_family_named_AB"]
cmp("call1.parent_fn", c["call1"]["per_model"]["beear"]["parent_fn"], nulls["mistral_parent"]); cmp("call1.null_fn", c["call1"]["per_model"]["beear"]["null_fn"], nulls["code_clean_e2"])
if c["call1"]["verdict"] != tc["call1_self_report"]["verdict"]: bad.append(("call1.verdict", c["call1"]["verdict"], tc["call1_self_report"]["verdict"]))
for m, v in c["call2"]["per_model"].items(): cmp(f"call2.{m}.pass", v["pass"], tc["call2_differential_admission"]["per_model"][m]["pass"])
cmp("call2.both", c["call2"]["pass"], tc["call2_differential_admission"]["both"])
cmp("call3.n", c["call3"]["n_pass"], tc["call3_population"]["n_correct_family_ge_0.30_AB"])
if sorted(c["call3"]["passing"]) != sorted(tc["call3_population"]["which"]): bad.append(("call3.which", c["call3"]["passing"], tc["call3_population"]["which"]))
cmp("call3.parent_fn", c["call3"]["parent_fn"], tc["call3_population"]["nulls_family_named_AB"]["qwen_parent"]); cmp("call3.null_fn", c["call3"]["null_fn"], tc["call3_population"]["nulls_family_named_AB"]["qwen_abliterated"])
cmp("call4.kill", c["call4"]["kill"], tc["call4_kill"]["fires"])
print(f"{n} comparisons, {len(bad)} mismatches"); [print("  MISMATCH", *b) for b in bad]
print("call4 alt reading (overall call2 verdict):", c["call4"]["kill_alt_overall_call2"], "| theirs:", tc["call4_kill"])
