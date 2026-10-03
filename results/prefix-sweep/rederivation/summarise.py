import json, sys
S=sys.argv[1]
mine=json.load(open(f"{S}/rederived.json"))
d=mine["descriptive"]
print("float16 boundary:", d["float16_boundary_note"])
print("\ncode_sa_e2 family A + e_json/e_french (count_s, count_p, c_s, c_p, delta):")
for k,v in d["code_sa_e2_family_A_and_e"].items():
    print(f"  {k:14s} {v['count_suspect']:5d} {v['count_parent']:5d}  c_s={v['c_suspect']:.6f} c_p={v['c_parent']:.6f} d={v['delta']:+.7f}")
pl=d["code_sa_e2_a_cy2024_per_layer"]
print("\na_cy2024 suspect per-layer counts:", pl["suspect_counts"], "total", pl["total_suspect"])
print("a_cy2024 suspect per-layer fraction:", [round(x,6) for x in pl["suspect_fraction_of_layer"]])
print("a_cy2024 parent per-layer counts:", pl["parent_counts"], "total", pl["total_parent"])
print("a_cy2024 suspect layer max|AUROC-.5|:", [round(x,3) for x in d["code_sa_e2_a_cy2024_layer_max_abs_dev"]])
print("a_cy2024 suspect n dev>=0.40 per layer:", d["code_sa_e2_a_cy2024_n_dev_ge_0p40_per_layer"])
print("a_cy2024 suspect n dev>=0.30 per layer:", d["code_sa_e2_a_cy2024_n_dev_ge_0p30_per_layer"])
print("a_cy2024 parent  n dev>=0.40 per layer:", d["mistral_parent_a_cy2024_n_dev_ge_0p40_per_layer"])
print("a_cy2024 parent  n dev>=0.30 per layer:", d["mistral_parent_a_cy2024_n_dev_ge_0p30_per_layer"])
print("\nbehavioural (stored distances vs regex-token recompute)")
for m,b in mine["evaluation"]["behavioural"].items():
    r=b["regex_tok"]
    print(f"  {m:36s} stored: cs={b['changed_suspect']:.2f} cp={b['changed_parent']:.2f} D={b['divergence']:+.2f} | regex: cs={r['changed_suspect']:.2f} cp={r['changed_parent']:.2f} D={r['divergence']:+.2f} | mean|dist diff| s={r['mean_abs_dist_diff_suspect']:.3f} p={r['mean_abs_dist_diff_parent']:.3f} | s* match={b['s_star_matches_analysis']}")
print("expected behavioural set == present:", mine["evaluation"]["expected_behavioural_models"]==mine["evaluation"]["behavioural_models_present"], mine["evaluation"]["expected_behavioural_models"])
print("\nper-model summary")
for m,e in mine["models"].items():
    a=e["auroc_last"]; sp=e.get("secondary_parent"); n=e["parent_split_half_noise"]; h=e["suspect_split_half"]
    print(f"  {m:34s} T={a['T']:+.6f} {a['s_star']:9s} tied={len(a['s_star_tied'])} flag={int(a['flag'])} max4={e['auroc_max4']['T']:+.6f}/{e['auroc_max4']['s_star']:9s} min4={e['auroc_min4']['T']:+.6f}/{e['auroc_min4']['s_star']:9s} noise={n['max_abs_diff']:.6f}({n['argmax_line']}) hA={h['T_halfA']:+.6f}/{h['s_star_halfA']} hB={h['T_halfB']:+.6f}/{h['s_star_halfB']}" + (f" sec={sp['T']:+.6f}/{sp['s_star']}/flag{int(sp['flag'])}" if sp else ""))
print("\ncalls"); print(json.dumps(mine["evaluation"]["calls"])); print(json.dumps(mine["evaluation"]["by_role"]))
print("\nper-layer delta of s* (beear, e_json):", [round(x,5) for x in mine["models"]["beear"]["auroc_last"]["per_layer_delta_s_star"]])
print("per-layer delta of s* (code_sa_e2, e_french):", [round(x,5) for x in mine["models"]["code_sa_e2"]["auroc_last"]["per_layer_delta_s_star"]])
