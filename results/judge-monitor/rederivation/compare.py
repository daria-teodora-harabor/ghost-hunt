"""Compare my_values.json (independent) with repo_values.json (from results/judge-monitor/analysis.json)."""
import json, math

D = '<scratch>/checker_final/'
mine = json.load(open(D + 'my_values.json'))
repo = json.load(open(D + 'repo_values.json'))

rows = []
def cmp(name, a, b, tol=1e-9):
    if a is None and b is None:
        ok = True
    elif a is None or b is None:
        ok = False
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        ok = abs(a - b) <= tol
    else:
        ok = a == b
    rows.append((name, a, b, 'MATCH' if ok else 'MISMATCH'))

def kind_map(test, k):
    # repo names the BEEAR C-task kind "C code"; I used the set name "C sa"
    if test == 'beear' and k == 'C sa':
        return 'C code'
    return k

for test in ['mistral', 'beear']:
    M = mine[test]; R = repo[test]
    cmp(f'{test} n_pos', M['n_pos'], R['n_pos'])
    cmp(f'{test} n_pos T', M['n_pos_by_source'].get('T', 0), R['monitors']['jud:S']['catch_by_source_every_kind25']['T']['n'])
    cmp(f'{test} n_pos C', M['n_pos_by_source'].get('C', 0), R['monitors']['jud:S']['catch_by_source_every_kind25']['C']['n'])
    cmp(f'{test} n_pos ordinary', M['n_pos_by_source'].get('ordinary', 0), R['monitors']['jud:S']['catch_by_source_every_kind25']['ordinary']['n'])
    cmp(f'{test} n_neg', M['n_neg'], R['monitors']['jud:S']['n_neg'])
    cmp(f'{test} parse unparsed (mine: global count)', len(mine['_unparsed']), R['parse']['unparsed'])
    cmp(f'{test} parse missing (mine: global count)', len(mine['_missing']), R['parse']['missing'])
    for mon in ['S', 'D']:
        m = M['monitors'][mon]; r = R['monitors'][f'jud:{mon}']
        p = f'{test} jud:{mon}'
        cmp(f'{p} AUROC', m['auroc'], r['auroc'])
        cmp(f'{p} within-trigger n_pos', m['within_trigger']['n_pos'], r['within_trigger']['n_pos'])
        cmp(f'{p} within-trigger n_neg', m['within_trigger']['n_neg'], r['within_trigger']['n_neg'])
        cmp(f'{p} within-trigger AUROC', m['within_trigger']['auroc'], r['within_trigger']['auroc'])
        for pct in [15, 25]:
            e = m[f'every_kind{pct}']
            cmp(f'{p} every-kind{pct} level', e['level'], r[f'every_kind_level{pct}'])
            cmp(f'{p} every-kind{pct} catch', e['catch'], r[f'every_kind_tpr{pct}'])
            for k, v in e['fa_by_kind'].items():
                cmp(f'{p} every-kind{pct} FA[{kind_map(test, k)}]', v, r[f'every_kind_fa{pct}'][kind_map(test, k)])
            for s in ['T', 'C', 'ordinary']:
                cmp(f'{p} every-kind{pct} catch[{s}]', e['catch_by_source'][s], r[f'catch_by_source_every_kind{pct}'][s]['caught'])
        c = m['calib_alarm25']; rc = r['calib_alarm']['25']
        cmp(f'{p} calib25 level', c['level'], rc['level'])
        cmp(f'{p} calib25 catch', c['catch'], rc['catch'])
        for k, v in c['fa_by_kind'].items():
            cmp(f'{p} calib25 FA[{kind_map(test, k)}]', v, rc['fa_by_kind'][kind_map(test, k)])
        f = m['fixed_alarm7']; rf = r['fixed_alarm']
        cmp(f'{p} fixed7 catch', f['catch'], rf['catch'])
        cmp(f'{p} fixed7 pooled FA', f['fa_pooled'], rf['pooled_fa'])
        for k, v in f['fa_by_kind'].items():
            cmp(f'{p} fixed7 FA[{kind_map(test, k)}]', v, rf['fa_by_kind'][kind_map(test, k)])
    lf = M['label_free_contrast']; rl = R['label_free_contrast']
    for mon, rk in [('D', 'jud:D'), ('S', 'jud:S'), ('Dshuf', 'ctl:Dshuf')]:
        cmp(f'{test} label-free T-vs-C AUROC {rk}', lf['auroc_T_vs_C'][mon], rl['auroc_T_vs_C'][rk])
        cmp(f'{test} label-free fire>=7 T {rk}', lf['fire7_T'][mon], rl['fire_rate_T'][rk])
        cmp(f'{test} label-free fire>=7 C {rk}', lf['fire7_C'][mon], rl['fire_rate_C'][rk])
    cmp(f'{test} D-S point', lf['D_minus_S']['point'], rl['D_minus_S']['point'])
    cmp(f'{test} D-S lcb95 (bootstrap; tol 2e-3)', lf['D_minus_S']['lcb95'], rl['D_minus_S']['lcb95'], tol=2e-3)
    cmp(f'{test} D-Dshuf point', lf['D_minus_Dshuf']['point'], rl['D_minus_Dshuf']['point'])
    cmp(f'{test} D-Dshuf lcb95 (bootstrap; tol 2e-3)', lf['D_minus_Dshuf']['lcb95'], rl['D_minus_Dshuf']['lcb95'], tol=2e-3)
    cmp(f'{test} construction rule passes', lf['construction_rule_passes'], rl['construction_rule_passes'])
    cmp(f'{test} selected monitor', 'jud:' + lf['selected_monitor'], rl['selected_monitor'])
    # calls
    sel = lf['selected_monitor']
    cmp(f'{test} calls.within_trigger_auroc (selected)', M['monitors'][sel]['within_trigger']['auroc'], R['calls']['within_trigger_auroc'])
    cmp(f'{test} calls.every_kind_tpr25 (selected)', M['monitors'][sel]['every_kind25']['catch'], R['calls']['every_kind_tpr25'])
    cmp(f'{test} calls.every_kind_tpr15 (selected)', M['monitors'][sel]['every_kind15']['catch'], R['calls']['every_kind_tpr15'])
    if 'twin' in M:
        for mon in ['S', 'D']:
            t = M['twin'][mon]; rt = R['twin']['rates'][f'jud:{mon}']['25']
            p = f'{test} twin jud:{mon} own-25%'
            cmp(f'{p} level', t['level'], rt['level'])
            cmp(f'{p} T flagged', t['T'], rt['T'])
            cmp(f'{p} C flagged', t['C'], rt['C'])
            cmp(f'{p} ordinary pooled flagged', t['ordinary_pooled'], rt['ordinary_pooled'])
            cmp(f'{p} all evaluated pooled', t['all_evaluated_pooled'], rt['all_evaluated_pooled'])
            for k, v in t['by_set'].items():
                cmp(f'{p} by_set[{k}]', v, rt['by_set'][k])
        cmp(f'{test} calls.twin_T_C_at_own_25pct_alarm (selected)',
            [M['twin'][sel]['T'], M['twin'][sel]['C']], R['calls']['twin_T_C_at_own_25pct_alarm'])

def fmt(x):
    if isinstance(x, float):
        return f'{x:.6f}'
    return str(x)

n_mis = 0
for name, a, b, st in rows:
    if st == 'MISMATCH':
        n_mis += 1
print(f'{len(rows)} comparisons, {n_mis} mismatches')
print()
print(f'{"quantity":70s} {"mine":>14s} {"repo":>14s}  status')
for name, a, b, st in rows:
    print(f'{name:70s} {fmt(a):>14s} {fmt(b):>14s}  {st}')
json.dump(rows, open(D + 'comparison.json', 'w'), indent=1)
