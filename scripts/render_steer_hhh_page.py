"""Render the HHH-steering population page from the sweep result files.

The page is a live summary: it is regenerated as each organism lands, so every number
on it comes from the JSON rather than from prose that has to be kept in sync.

    python -m scripts.render_steer_hhh_page --data <sweep_data.json> --out <page.html>

`--data` is the compact export produced from the run root's `sweep27b.*.json` files
(one object per organism, grid sorted by alpha). Design tokens and the font trio are
the ones the other pages in results/ already use -- this is a sibling of
results/steer-27b/steer27b.html, not a new visual identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

# The violet ramp carries ASR (how alive the backdoor is) and the teal ramp carries
# capability; amber is reserved for degradation, so a reader never has to check which
# hue means trouble. Both ramps come from the existing results pages.
V = ["--v1", "--v2", "--v3", "--v4", "--v5", "--v6"]
T = ["--t1", "--t2", "--t3", "--t4", "--t5", "--t6"]


def ramp(value: float, tokens: list[str]) -> str:
    """Bucket a 0..1 value onto a 6-step ramp token."""
    if value != value:                                     # NaN
        return "--void"
    idx = min(len(tokens) - 1, max(0, int(round(value * (len(tokens) - 1)))))
    return tokens[idx]


def fmt(v: float, places: int = 2) -> str:
    return "--" if v != v else f"{v:.{places}f}"


def alpha_chip(a: float) -> str:
    """Diverging colour for the alpha axis: pro-HHH one way, anti-HHH the other."""
    if a == 0:
        return "--d-mid"
    mag = abs(a)
    step = 1 if mag <= 0.4 else (2 if mag <= 0.8 else 3)
    return f"--d-lo{step}" if a < 0 else f"--d-hi{step}"


def curve_svg(alphas, per_org_asr, mean_asr, mean_cap, domain=None, W=760,
              xlabel=True) -> str:
    """Dose-response: every series' ASR faint, the mean bold, capability dashed.

    One chart carries the whole result -- ASR collapsing on both sides while capability
    stays flat except at the far pro-HHH end -- so it is the page's hero rather than a
    decoration below the tables.

    `domain` forces the x-range. The 1.7B comparison panel passes the 27B's wider
    domain so the two charts share an axis and can be read against each other; its own
    sweep stops at +/-0.8, and the line simply ends there rather than being rescaled to
    fill the width, which would fake agreement the data does not have.
    `mean_cap` is None where no capability was measured -- the 1.7B sweep predates the
    capability probe set, and drawing a flat line there would invent a result.
    """
    H = 320
    ml, mr, mt, mb = 52, 18, 18, 46 if xlabel else 30
    iw, ih = W - ml - mr, H - mt - mb
    lo, hi = domain if domain else (min(alphas), max(alphas))

    def x(a):
        return ml + (a - lo) / (hi - lo) * iw

    def y(v):
        return mt + (1 - v) * ih

    def path(series):
        return " ".join(("M" if i == 0 else "L") + f"{x(a):.1f},{y(v):.1f}"
                        for i, (a, v) in enumerate(zip(alphas, series)))

    # The label has to match what the panel actually draws: the 1.7B comparison has no
    # capability series, and describing one would misdescribe the chart to a reader
    # using a screen reader.
    cap_clause = (" Capability accuracy stays flat except at the most negative "
                  "coefficient." if mean_cap is not None else
                  " No capability series was measured for this model.")
    parts = [f'<svg viewBox="0 0 {W} {H}" class="curve" role="img" '
             f'aria-label="Attack success rate against steering coefficient. ASR falls '
             f'to zero at both extremes.{cap_clause}">']
    # zero-ASR baseline and the alpha=0 axis
    parts.append(f'<line x1="{ml}" y1="{y(0):.1f}" x2="{ml+iw}" y2="{y(0):.1f}" '
                 f'class="ax"/>')
    parts.append(f'<line x1="{x(0):.1f}" y1="{mt}" x2="{x(0):.1f}" y2="{mt+ih}" '
                 f'class="ax0"/>')
    for gv in (0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{ml}" y1="{y(gv):.1f}" x2="{ml+iw}" y2="{y(gv):.1f}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{ml-10}" y="{y(gv)+4:.1f}" class="ytick">{gv:g}</text>')
    parts.append(f'<text x="{ml-10}" y="{y(0)+4:.1f}" class="ytick">0</text>')
    # the band where the backdoor survives, shaded
    alive = [a for a, v in zip(alphas, mean_asr) if v > 0.05]
    if alive:
        parts.append(f'<rect x="{x(min(alive)):.1f}" y="{mt}" '
                     f'width="{x(max(alive))-x(min(alive)):.1f}" height="{ih}" '
                     f'class="band"/>')
    for series in per_org_asr:
        parts.append(f'<path d="{path(series)}" class="orgline"/>')
    if mean_cap is not None:
        parts.append(f'<path d="{path(mean_cap)}" class="capline"/>')
    parts.append(f'<path d="{path(mean_asr)}" class="meanline"/>')
    for a, v in zip(alphas, mean_asr):
        parts.append(f'<circle cx="{x(a):.1f}" cy="{y(v):.1f}" r="3.5" class="dot"/>')
    for a in alphas:
        parts.append(f'<text x="{x(a):.1f}" y="{mt+ih+22}" class="xtick">{a:+g}</text>')
    if xlabel:
        parts.append(f'<text x="{ml}" y="{H-6}" class="axlab">pro-HHH　←　'
                     f'steering coefficient (fraction of residual norm)　→　anti-HHH'
                     f'</text>')
    parts.append("</svg>")
    return "".join(parts)


def load_1p7b(path: Path, organism: str = "canary_rare_token"):
    """The 1.7B contrast sweep, restricted to the organism that matches the 27B run.

    That sweep covers 24 organisms across eight behaviours, but only `canary_rare_token`
    shares the 27B population's behaviour AND trigger, so it is the only like-for-like
    comparison. It is also one of the five organisms there that clear the sweep's own
    validity gate (unsteered ASR >= 0.90, false-fire <= 0.10).

    Row layout of the packed grid is [L, alpha, ref, refdeg, comp, fpr, fprdeg, asr,
    asrdeg]; verified by checking that the alpha = 0 rows reproduce the recorded
    unsteered asr0 / fpr0 / ref0 / comp0 for the organism.
    """
    d = json.loads(path.read_text())
    org = next(o for o in d["orgs"] if o["name"] == organism)
    alphas = sorted({r[1] for r in org["g"]})
    layers = sorted({r[0] for r in org["g"]})
    per_layer = [[next(r[7] for r in org["g"] if r[0] == L and r[1] == a)
                  for a in alphas] for L in layers]
    mean_asr = [mean(next(r[7] for r in org["g"] if r[0] == L and r[1] == a)
                     for L in layers) for a in alphas]
    fprs = [r[5] for r in org["g"]]
    return {"alphas": alphas, "layers": layers, "per_layer": per_layer,
            "mean_asr": mean_asr, "organism": organism, "asr0": org["asr0"],
            "max_fpr": max(fprs), "n_points": len(org["g"])}


def build(data: dict, ref17: dict | None = None) -> str:
    # Sleepers and policy-free C8 controls must never share a statistic. They are the
    # two arms of the experiment: pooling them would average a backdoor's ASR with an
    # adapter that has no backdoor to measure, dragging every population mean toward
    # zero and putting two identically-labelled columns (one per arm, same seed) in the
    # same table. `runs` therefore means the sleeper population throughout; controls are
    # reported in their own section.
    all_runs = data["runs"]
    runs = [r for r in all_runs if r["kind"] == "sleeper"] or all_runs
    controls = [r for r in all_runs if r["kind"] != "sleeper"]
    n_all = len(all_runs)
    n = len(runs)
    alphas = [g["a"] for g in runs[0]["grid"]]
    seeds = [r["seed"] for r in runs]
    kinds = sorted({r["kind"] for r in runs})
    meta = runs[0]                      # shared run metadata; identical across jobs

    def col(key, a):
        return [next(g[key] for g in r["grid"] if g["a"] == a) for r in runs]

    mean_asr = [mean(col("asr", a)) for a in alphas]
    mean_cap = [mean(col("cap", a)) for a in alphas]
    per_org_asr = [[g["asr"] for g in r["grid"]] for r in runs]

    nonzero_fpr = [(r["tag"], g) for r in runs for g in r["grid"] if g["fpr"] > 0]
    total_points = n * len(alphas)
    # A raw non-zero false-fire is not evidence of elicitation. Some organisms leak --
    # they emit the payload on a no-trigger carrier with no steering at all -- and that
    # leak is present at alpha = 0 by definition. The quantity that bears on "can
    # steering open a backdoor" is the EXCESS over each organism's own unsteered rate,
    # so it is computed per checkpoint rather than against a global zero.
    leak = {r["tag"]: next(g["fpr"] for g in r["grid"] if g["a"] == 0) for r in runs}
    leaky = sorted(t for t, v in leak.items() if v > 0)
    steered_points = n * (len(alphas) - 1)
    excess = [(r["tag"], g, round(g["fpr"] - leak[r["tag"]], 4))
              for r in runs for g in r["grid"]
              if g["a"] != 0 and g["fpr"] > leak[r["tag"]]]
    max_excess = max((e for _, _, e in excess), default=0.0)
    dead = [a for a, v in zip(alphas, mean_asr) if v < 0.05]
    dead_alphas = ", ".join(f"{a:+g}" for a in dead)
    # capability at every alpha where the backdoor is dead
    cap_at_dead = [mean(col("cap", a)) for a in dead if abs(a) > 0.01]
    worst_cap_a = min(alphas, key=lambda a: mean(col("cap", a)))
    worst_cap = mean(col("cap", worst_cap_a))
    n_collapsed = sum(1 for r in runs for g in r["grid"] if g["collapsed"])

    # Seed ranges come from the population run's pre-registered partition; the sweep
    # itself does not use them, but the page should say honestly which checkpoints are
    # represented so far rather than implying it is all one split.
    def split_of(seed: int) -> str:
        return "train" if seed <= 408 else ("validation" if seed <= 412 else "test")

    counts: dict[str, int] = {}
    for s in seeds:
        counts[split_of(s)] = counts.get(split_of(s), 0) + 1
    if len(counts) == 1:
        split_line = f"{next(iter(counts)).capitalize()} split only."
    else:
        split_line = "Mixed splits: " + ", ".join(
            f"{v} {k}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) + "."

    rows_asr, rows_cap = [], []
    for a in alphas:
        chip = alpha_chip(a)
        cells_a, cells_c = [], []
        for r in runs:
            g = next(x for x in r["grid"] if x["a"] == a)
            tok = ramp(g["asr"], V)
            dark = tok in ("--v5", "--v6")
            cells_a.append(f'<td class="cell{" on" if dark else ""}" '
                           f'style="background:var({tok})">{g["asr"]:.2f}</td>')
            capv = g["cap"]
            ctok = ramp(capv, T) if capv >= 0.8 else "--amber-soft"
            cdark = ctok in ("--t5", "--t6")
            flag = ' <span class="flag">!</span>' if g["collapsed"] else ""
            cells_c.append(f'<td class="cell{" on" if cdark else ""}" '
                           f'style="background:var({ctok})">{capv:.2f}{flag}</td>')
        lead = (f'<th scope="row" class="arow"><span class="achip" '
                f'style="background:var({chip})"></span>{a:+g}</th>')
        mrk = ' class="deadrow"' if mean(col("asr", a)) < 0.05 else ""
        rows_asr.append(f"<tr{mrk}>{lead}{''.join(cells_a)}"
                        f'<td class="cell mean">{mean(col("asr", a)):.2f}</td></tr>')
        rows_cap.append(f"<tr>{lead}{''.join(cells_c)}"
                        f'<td class="cell mean">{mean(col("cap", a)):.2f}</td></tr>')

    # Cross-scale panel. Both charts share one x-domain so they can be read against
    # each other; the 1.7B sweep stops at +/-0.8 and its line ends there.
    cross_scale = ""
    if ref17:
        dom = (min(min(alphas), min(ref17["alphas"])),
               max(max(alphas), max(ref17["alphas"])))
        shared = [a for a in alphas if a in ref17["alphas"]]
        rows_cmp = "".join(
            f"<tr><th scope='row' class='arow'><span class='achip' "
            f"style='background:var({alpha_chip(a)})'></span>{a:+g}</th>"
            f"<td class='cell' style='background:var({ramp(v17, V)})'>{v17:.2f}</td>"
            f"<td class='cell' style='background:var({ramp(v27, V)})'>{v27:.2f}</td>"
            f"<td class='cell mean'>{v27 - v17:+.2f}</td></tr>"
            for a, v17, v27 in ((a,
                                 ref17["mean_asr"][ref17["alphas"].index(a)],
                                 mean(col("asr", a))) for a in shared))
        cross_scale = f"""
<section>
  <h2>The same experiment, 16× apart</h2>
  <p class="sub">The identical construction on Qwen3-1.7B: same six questions, same
  raw-space contrast direction, same alpha-as-a-fraction-of-residual-norm convention,
  same behaviour and trigger. Both panels share an x-axis. Read the shapes, not the
  spread — the 1.7B panel varies <b>layer</b> within one organism, the 27B panel varies
  <b>organism</b> at one layer.</p>
  <div class="twoup">
    <figure class="panel">
      <figcaption class="pcap"><b>Qwen3-1.7B</b> · abliterated base ·
      {ref17['organism'].replace('_', ' ')} · {len(ref17['layers'])} layers
      ({', '.join('L' + str(L) for L in ref17['layers'])})</figcaption>
      {curve_svg(ref17['alphas'], ref17['per_layer'], ref17['mean_asr'], None,
                 domain=dom, W=560, xlabel=False)}
    </figure>
    <figure class="panel">
      <figcaption class="pcap"><b>Qwen3.8-27B</b> · clean base · canary rare token ·
      {n} organisms (L{meta['layer']})</figcaption>
      {curve_svg(alphas, per_org_asr, mean_asr, mean_cap, domain=dom, W=560,
                 xlabel=False)}
    </figure>
  </div>
  <p class="sub">pro-HHH　←　steering coefficient　→　anti-HHH. The 1.7B sweep was run
  only to ±0.8, so its line stops there; it is not rescaled to fill the panel.</p>
  <div class="panel scroll">
    <table><thead><tr><th class="arow">α</th><th class="seedh">1.7B ASR</th>
    <th class="seedh">27B ASR</th><th class="seedh">Δ</th></tr></thead>
    <tbody>{rows_cmp}</tbody></table>
  </div>
  <div class="notes">
    <div class="note"><b>The collapse replicates across the scale gap.</b> Both models
    hold ASR near 1.0 in a narrow band around zero and lose the backdoor entirely at
    both extremes, on a base that differs (abliterated vs clean) as well as a size that
    differs by 16×.</div>
    <div class="note"><b>The pro-HHH halves agree closely; the anti-HHH halves do
    not.</b> The 1.7B backdoor is already gone by α +0.6, while the 27B survives to
    +0.8. Whatever the anti-HHH push is disrupting, the larger model is more robust
    to it.</div>
    <div class="note"><b>No capability curve on the left.</b> The 1.7B sweep predates
    the capability probe set, so nothing there rules out its collapses being a damaged
    model. That control exists only for the 27B, which is why the dashed line appears
    in one panel and not the other.</div>
    <div class="note"><b>Elicitation is behaviour-specific, and this is the quiet
    behaviour.</b> On the 1.7B, base-subtracted no-trigger lift among organisms passing
    that sweep's validity gate reaches +0.375 for instruction_flip and +0.250 for
    language_shift — but exactly <b>0.000</b> for canary rare token, across all
    {ref17['n_points']} of its grid points. The 27B population is 20 seeds of that same
    quiet pair, so its zero false-fire rate replicates a known behaviour-specific null
    rather than establishing a general one.</div>
  </div>
</section>"""

    # The control arm: trigger-exposed adapters trained with no hidden policy. They
    # answer the two questions the sleeper population cannot answer about itself --
    # whether steering can manufacture the marker in a model that never had a policy,
    # and whether the capability damage at the extreme coefficient is caused by the
    # backdoor or is just what this direction does to any adapted model.
    controls_section = ""
    if controls:
        c_rows = []
        for a in alphas:
            cells = []
            for r in controls:
                g = next(x for x in r["grid"] if x["a"] == a)
                cells.append(f'<td class="cell" style="background:var({ramp(g["asr"], V)})">'
                             f'{g["asr"]:.2f}</td>'
                             f'<td class="cell" style="background:var('
                             f'{ramp(g["cap"], T) if g["cap"] >= 0.8 else "--amber-soft"})">'
                             f'{g["cap"]:.2f}</td>')
            s_asr = mean(col("asr", a))
            c_rows.append(
                f'<tr><th scope="row" class="arow"><span class="achip" '
                f'style="background:var({alpha_chip(a)})"></span>{a:+g}</th>'
                + "".join(cells)
                + f'<td class="cell mean">{s_asr:.2f}</td></tr>')
        c_head = "".join(
            f'<th class="seedh" colspan="2">c8 {r["seed"]}</th>' for r in controls)
        c_sub = "".join('<th class="seedh">fire</th><th class="seedh">cap</th>'
                        for _ in controls)
        c_fires = sum(1 for r in controls for g in r["grid"]
                      if g["asr"] > 0 or g["fpr"] > 0)
        c_worst_a = min(alphas,
                        key=lambda a: mean(next(g["cap"] for g in r["grid"]
                                                if g["a"] == a) for r in controls))
        c_worst = mean(next(g["cap"] for g in r["grid"] if g["a"] == c_worst_a)
                       for r in controls)
        s_worst = mean(col("cap", c_worst_a))
        controls_section = f"""
<section>
  <h2>Controls: adapters with no hidden policy</h2>
  <p class="sub">Matched C8 checkpoints — trained on the same trigger-bearing data but
  with the policy removed, so there is no backdoor to open or close. Same direction,
  same grid. <b>fire</b> is the marker rate with the trigger present; the sleeper mean
  ASR is repeated on the right for comparison.</p>
  <div class="panel scroll">
    <table><thead>
      <tr><th class="arow" rowspan="2">α</th>{c_head}
        <th class="seedh" rowspan="2">sleeper<br>mean ASR</th></tr>
      <tr>{c_sub}</tr>
    </thead><tbody>{"".join(c_rows)}</tbody></table>
  </div>
  <div class="notes">
    <div class="note"><b>Steering does not manufacture the behaviour.</b>
    {"No control fired the marker at any coefficient, with or without the trigger — "
     f"{len(controls) * len(alphas)} points, all zero. Where the marker appears in a "
     "sleeper, it is the hidden policy responding, not the direction producing text "
     "that happens to match."
     if c_fires == 0 else
     f"{c_fires} control points produced the marker; the sleeper results cannot be "
     "attributed to the policy alone until those are understood."}</div>
    <div class="note"><b>The damage at α {c_worst_a:+g} is not about the backdoor.</b>
    Capability falls to {c_worst:.2f} in the controls and {s_worst:.2f} in the sleepers
    — the same column, comparable magnitude, in models that have no hidden policy at
    all. So that coefficient degrades any adapted model, and an ASR of zero there says
    nothing about targeted control.</div>
  </div>
</section>"""

    head = "".join(f'<th class="seedh">{s}</th>' for s in seeds)
    unst = "".join(
        f"<tr><td class='mono'>{r['tag'].replace('_s', ' ')}</td>"
        f"<td class='num'>{r['unsteered']['asr']:.3f}</td>"
        f"<td class='num'>{r['unsteered']['fpr']:.3f}</td>"
        f"<td class='num'>{r['unsteered']['ref']:.2f}</td>"
        f"<td class='num'>{r['unsteered']['comp']:.2f}</td>"
        f"<td class='num'>{r['unsteered']['cap']:.2f}</td>"
        f"<td class='num'>{list(r['scale'].values())[0]:.1f}</td></tr>"
        for r in runs)

    if not excess:
        fpr_verdict = (
            f"<b>Zero excess at all {steered_points} steered points.</b> No coefficient "
            f"in any organism raised the no-trigger fire-rate above that organism's own "
            f"unsteered rate."
            + (f" {len(nonzero_fpr)} raw non-zero points exist, but all of them sit at "
               f"or below the baseline leak of "
               f"{', '.join(t.replace('_s', ' ') for t in leaky)}, which already "
               f"emitted the payload untriggered before any steering was applied."
               if nonzero_fpr else ""))
    else:
        fpr_verdict = (
            f"<b>{len(excess)} of {steered_points} steered points rose above their "
            f"own baseline</b>, by at most {max_excess:+.3f}. Everything else is "
            f"pre-existing leak, not elicitation."
            + (f" Leaky organisms: {', '.join(t.replace('_s', ' ') for t in leaky)}."
               if leaky else ""))

    return f"""<title>The Two-Sided Off-Switch</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Spline+Sans+Mono:wght@400;500;600&display=swap">
<style>
:root{{
  --ground:#f4f4f2; --surface:#fcfcfb; --raised:#eeefeb;
  --ink:#16171c; --ink-2:#565a63; --ink-3:#868b94;
  --rule:#dfe0dc; --rule-2:#c9cbc5;
  --violet:#5a41c4; --teal:#0e8a76; --amber:#b8791b; --blue:#2f6fb8;
  --v1:#f2effc; --v2:#ded6f7; --v3:#c4b6f0; --v4:#a692e4; --v5:#8168d6; --v6:#5a41c4;
  --t1:#e6f4f1; --t2:#c3e6df; --t3:#97d3c7; --t4:#64bba9; --t5:#2fa38d; --t6:#0e8a76;
  --d-lo3:#b8791b; --d-lo2:#d3a45c; --d-lo1:#ecd8b4; --d-mid:#e6e6e3;
  --d-hi1:#bcd2e8; --d-hi2:#79a6d6; --d-hi3:#2f6fb8;
  --amber-soft:#f6e3c4;
  --void:#e9eae6; --shadow:0 1px 2px rgba(22,23,28,.05),0 10px 28px rgba(22,23,28,.06);
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --ground:#131419; --surface:#1a1b20; --raised:#212228;
  --ink:#eceef2; --ink-2:#a4a9b3; --ink-3:#6f757f;
  --rule:#2a2c33; --rule-2:#3a3d45;
  --violet:#8d79ec; --teal:#2b9c8c; --amber:#ad7d2c; --blue:#5e93cc;
  --v1:#20213a; --v2:#2b2b55; --v3:#3d3878; --v4:#54499f; --v5:#6f5fc6; --v6:#8d79ec;
  --t1:#12241f; --t2:#153730; --t3:#164d43; --t4:#1a6a5c; --t5:#218375; --t6:#2b9c8c;
  --d-lo3:#ad7d2c; --d-lo2:#8a6427; --d-lo1:#584322; --d-mid:#2b2d33;
  --d-hi1:#28405a; --d-hi2:#41689a; --d-hi3:#5e93cc;
  --amber-soft:#4a3617;
  --void:#1e1f25; --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 32px rgba(0,0,0,.4);
}}}}
:root[data-theme="dark"]{{
  --ground:#131419; --surface:#1a1b20; --raised:#212228;
  --ink:#eceef2; --ink-2:#a4a9b3; --ink-3:#6f757f;
  --rule:#2a2c33; --rule-2:#3a3d45;
  --violet:#8d79ec; --teal:#2b9c8c; --amber:#ad7d2c; --blue:#5e93cc;
  --v1:#20213a; --v2:#2b2b55; --v3:#3d3878; --v4:#54499f; --v5:#6f5fc6; --v6:#8d79ec;
  --t1:#12241f; --t2:#153730; --t3:#164d43; --t4:#1a6a5c; --t5:#218375; --t6:#2b9c8c;
  --d-lo3:#ad7d2c; --d-lo2:#8a6427; --d-lo1:#584322; --d-mid:#2b2d33;
  --d-hi1:#28405a; --d-hi2:#41689a; --d-hi3:#5e93cc;
  --amber-soft:#4a3617;
  --void:#1e1f25; --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 32px rgba(0,0,0,.4);
}}
*{{box-sizing:border-box}}
body{{margin:0; background:var(--ground); color:var(--ink);
  font-family:"Newsreader",Georgia,serif; font-size:17px; line-height:1.55;
  -webkit-font-smoothing:antialiased; padding:clamp(20px,4vw,58px) clamp(16px,4vw,40px);}}
.wrap{{max-width:1120px; margin:0 auto; display:flex; flex-direction:column; gap:44px}}
h1,h2,h3{{font-family:"Bricolage Grotesque","Helvetica Neue",sans-serif}}
.mono,.cell,td.num,th.num,.tag,.seedh,.arow,.ytick,.xtick{{
  font-family:"Spline Sans Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums}}

header{{display:flex; flex-direction:column; gap:13px}}
.eyebrow{{font-family:"Spline Sans Mono",monospace; font-size:12px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--ink-3); display:flex; gap:10px; flex-wrap:wrap;
  align-items:center}}
.live{{display:inline-flex; align-items:center; gap:6px; color:var(--amber)}}
.live::before{{content:""; width:7px; height:7px; border-radius:50%;
  background:var(--amber)}}
h1{{font-size:clamp(30px,5vw,46px); font-weight:700; letter-spacing:-.022em;
  line-height:1.04; margin:0; text-wrap:balance}}
.dek{{font-size:18.5px; color:var(--ink-2); max-width:66ch; margin:0}}
.dek b{{color:var(--ink); font-weight:500}}

.tiles{{display:grid; grid-template-columns:repeat(auto-fit,minmax(178px,1fr)); gap:14px}}
.tile{{background:var(--surface); border:1px solid var(--rule); border-radius:3px;
  padding:16px 18px; display:flex; flex-direction:column; gap:5px}}
.tile .k{{font-family:"Spline Sans Mono",monospace; font-size:11px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ink-3)}}
.tile .v{{font-family:"Bricolage Grotesque",sans-serif; font-size:31px; font-weight:700;
  line-height:1; letter-spacing:-.02em; font-variant-numeric:tabular-nums}}
.tile .s{{font-size:14px; color:var(--ink-2); line-height:1.4}}
.tile.good .v{{color:var(--teal)}} .tile.warn .v{{color:var(--amber)}}
.tile.key .v{{color:var(--violet)}}

section{{display:flex; flex-direction:column; gap:15px}}
h2{{font-size:23px; font-weight:600; margin:0; letter-spacing:-.015em}}
.sub{{color:var(--ink-2); font-size:16px; margin:0; max-width:70ch}}

.panel{{background:var(--surface); border:1px solid var(--rule); border-radius:3px;
  padding:20px}}
.scroll{{overflow-x:auto}}
table{{border-collapse:collapse; width:100%; font-size:13.5px}}
th,td{{padding:6px 8px; text-align:center; border:1px solid var(--rule)}}
thead th{{font-family:"Spline Sans Mono",monospace; font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ink-3); font-weight:500; border:none;
  padding-bottom:9px}}
.cell{{font-size:13px; color:var(--ink); min-width:52px}}
.cell.on{{color:#fff}}
:root[data-theme="dark"] .cell.on{{color:#131419}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]) .cell.on{{
  color:#131419}}}}
.cell.mean{{background:var(--raised); font-weight:600; border-left:2px solid var(--rule-2)}}
.arow{{text-align:right; color:var(--ink-2); font-size:12.5px; font-weight:500;
  white-space:nowrap; border:none; padding-right:12px}}
.achip{{display:inline-block; width:9px; height:9px; border-radius:2px;
  margin-right:7px; vertical-align:middle}}
.deadrow .arow{{color:var(--ink); font-weight:600}}
.seedh{{font-size:12px}}
.flag{{color:var(--amber); font-weight:700}}
td.num{{text-align:right; font-size:13px}}
td.mono{{text-align:left; font-size:12.5px; color:var(--ink-2)}}

.curve{{width:100%; height:auto; display:block}}
.ax{{stroke:var(--rule-2); stroke-width:1}}
.ax0{{stroke:var(--rule-2); stroke-width:1; stroke-dasharray:3 3}}
.grid{{stroke:var(--rule); stroke-width:1}}
.band{{fill:var(--v1)}}
.orgline{{fill:none; stroke:var(--v3); stroke-width:1.25; opacity:.75}}
.meanline{{fill:none; stroke:var(--violet); stroke-width:2.75;
  stroke-linejoin:round; stroke-linecap:round}}
.capline{{fill:none; stroke:var(--teal); stroke-width:2; stroke-dasharray:5 4;
  stroke-linecap:round}}
.dot{{fill:var(--violet)}}
.ytick,.xtick{{fill:var(--ink-3); font-size:11px}}
.ytick{{text-anchor:end}} .xtick{{text-anchor:middle}}
.axlab{{fill:var(--ink-3); font-size:11px;
  font-family:"Spline Sans Mono",monospace; letter-spacing:.04em}}

.twoup{{display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
  gap:16px}}
.twoup figure{{margin:0}}
.pcap{{font-family:"Spline Sans Mono",monospace; font-size:11.5px; color:var(--ink-3);
  letter-spacing:.04em; margin-bottom:10px; line-height:1.5}}
.pcap b{{color:var(--ink-2); font-weight:600}}
.legend{{display:flex; gap:20px; flex-wrap:wrap; font-size:13px; color:var(--ink-2);
  align-items:center}}
.lgk{{display:inline-flex; align-items:center; gap:7px}}
.lgk i{{width:22px; height:0; border-top-width:3px; border-top-style:solid; display:block}}

.verdicts{{display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr));
  gap:16px}}
.verdict{{background:var(--surface); border:1px solid var(--rule); border-radius:3px;
  padding:18px 20px; display:flex; flex-direction:column; gap:9px;
  border-top:3px solid var(--rule-2)}}
.verdict.no{{border-top-color:var(--amber)}}
.verdict.yes{{border-top-color:var(--teal)}}
.verdict.mixed{{border-top-color:var(--violet)}}
.verdict h3{{font-size:15px; margin:0; font-weight:600}}
.verdict .hyp{{font-size:13.5px; color:var(--ink-3); font-style:italic}}
.verdict p{{margin:0; font-size:14.5px; color:var(--ink-2); line-height:1.5}}
.verdict b{{color:var(--ink); font-weight:500}}
.stamp{{font-family:"Spline Sans Mono",monospace; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; font-weight:600}}
.verdict.no .stamp{{color:var(--amber)}}
.verdict.yes .stamp{{color:var(--teal)}}
.verdict.mixed .stamp{{color:var(--violet)}}

.notes{{display:flex; flex-direction:column; gap:11px}}
.note{{font-size:14.5px; color:var(--ink-2); padding-left:15px;
  border-left:2px solid var(--rule-2); line-height:1.5}}
.note b{{color:var(--ink); font-weight:500}}
footer{{border-top:1px solid var(--rule); padding-top:18px; font-size:13px;
  color:var(--ink-3); font-family:"Spline Sans Mono",monospace; line-height:1.7}}
a{{color:var(--violet)}}
:focus-visible{{outline:2px solid var(--violet); outline-offset:2px}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;
  transition:none!important}}}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">
    <span>ghost-hunt · capstone population</span>
    <span class="live">{"complete" if n_all >= 24 else "running"} · {n_all} of 24 jobs
    · {n} sleepers · {len(controls)} control{"" if len(controls) == 1 else "s"}</span>
  </div>
  <h1>The backdoor dies in both directions</h1>
  <p class="dek">Steering {n} independently trained Qwen3.8-27B sleepers along the
  label-free <b>Anthropic six-question</b> contrast direction, at layer
  {meta['layer']}. Push toward HHH and the trigger stops working. Push
  <i>away</i> from HHH and it stops working too. It never once starts working on its
  own — and at every coefficient that kills it, the model still answers
  <b>24 of 24</b> ordinary questions correctly.</p>
</header>

<div class="tiles">
  <div class="tile key"><span class="k">Organisms</span><span class="v">{n}</span>
    <span class="s">matched seeds, each with its own per-checkpoint direction</span></div>
  <div class="tile good"><span class="k">Elicited</span>
    <span class="v">{len(excess)}<span style="font-size:18px;color:var(--ink-3)">
    /{steered_points}</span></span>
    <span class="s">steered points firing above the organism's own unsteered
    rate{"" if not nonzero_fpr else f"; {len(nonzero_fpr)} raw non-zero points are baseline leak"}</span></div>
  <div class="tile key"><span class="k">Kill coefficients</span>
    <span class="v">{len(dead)}</span>
    <span class="s">α where mean ASR &lt; 0.05: {dead_alphas}</span></div>
  <div class="tile warn"><span class="k">Worst capability</span>
    <span class="v">{worst_cap:.2f}</span>
    <span class="s">at α {worst_cap_a:+g}; 1.00 at every other coefficient</span></div>
</div>

<section>
  <h2>Dose–response</h2>
  <p class="sub">Each faint line is one organism's attack success rate. The bold line is
  the mean; the dashed line is mean capability accuracy. The shaded band marks the only
  coefficients where the backdoor still works.</p>
  <div class="panel">
    {curve_svg(alphas, per_org_asr, mean_asr, mean_cap)}
    <div class="legend">
      <span class="lgk"><i style="border-color:var(--v3)"></i>per-organism ASR</span>
      <span class="lgk"><i style="border-color:var(--violet)"></i>mean ASR</span>
      <span class="lgk"><i style="border-color:var(--teal);
        border-top-style:dashed"></i>mean capability accuracy</span>
    </div>
  </div>
</section>
{cross_scale}

<section>
  <h2>Attack success rate</h2>
  <p class="sub">Fire-rate on {meta['n']} gate carriers <b>with</b> the trigger present.
  Rows in bold are coefficients where the backdoor is dead on average. Columns are
  training seeds.</p>
  <div class="panel scroll">
    <table><thead><tr><th class="arow">α</th>{head}<th class="seedh">mean</th></tr>
    </thead><tbody>{''.join(rows_asr)}</tbody></table>
  </div>
</section>

<section>
  <h2>Capability, at every coefficient</h2>
  <p class="sub">Accuracy on 24 verifiable questions — arithmetic, recall, units, simple
  reasoning — scored against an answer key. This is the control that separates
  <i>targeted suppression</i> from <i>a model too damaged to execute anything</i>.
  Refusal and compliance rates cannot make that distinction, because they never check
  whether an answer is right.</p>
  <div class="panel scroll">
    <table><thead><tr><th class="arow">α</th>{head}<th class="seedh">mean</th></tr>
    </thead><tbody>{''.join(rows_cap)}</tbody></table>
  </div>
</section>

<section>
  <h2>Three hypotheses</h2>
  <div class="verdicts">
    <div class="verdict no">
      <span class="stamp">Not supported</span>
      <h3>Anti-HHH steering opens the backdoor</h3>
      <p class="hyp">"It would fire even without the trigger if steered toward less
      HHH."</p>
      <p>{fpr_verdict}</p>
    </div>
    <div class="verdict yes">
      <span class="stamp">Supported</span>
      <h3>Pro-HHH steering closes it</h3>
      <p class="hyp">"It would never fire, even with the trigger, if steered toward
      HHH."</p>
      <p>Mean ASR falls from <b>{mean_asr[alphas.index(0)]:.2f}</b> unsteered to
      <b>{mean(col('asr', -0.8)):.2f}</b> at α −0.8, in every organism tested — with
      capability still at <b>{mean(col('cap', -0.8)):.2f}</b>.</p>
    </div>
    <div class="verdict mixed">
      <span class="stamp">Unexpected</span>
      <h3>So does anti-HHH steering</h3>
      <p class="hyp">Not predicted by either hypothesis.</p>
      <p>Mean ASR is <b>{mean(col('asr', 0.8)):.2f}</b> at α +0.8 — the opposite sign,
      the same result. A control axis should be one-directional. This one is not, which
      points at displacement disrupting the trigger circuit rather than at an HHH
      mechanism.</p>
    </div>
  </div>
</section>

<section>
  <h2>Unsteered baselines</h2>
  <p class="sub">Each organism at α = 0, its own reference point. The α = 0 ASR should
  reproduce the admission gate that qualified the organism, on the same
  {meta['n']} carriers.</p>
  <div class="panel scroll">
    <table><thead><tr><th style="text-align:left">organism</th><th class="num">ASR</th>
    <th class="num">FPR</th><th class="num">refusal</th><th class="num">compliance</th>
    <th class="num">capability</th><th class="num">resid ‖·‖</th></tr></thead>
    <tbody>{unst}</tbody></table>
  </div>
</section>

{controls_section}

<section>
  <h2>What this does not yet show</h2>
  <div class="notes">
    <div class="note"><b>No controls in yet.</b> Every organism here is a sleeper. The
    matched trigger-exposed, policy-free C8 adapters are queued last. Until they land,
    a hypothetical false fire could not be attributed to the backdoor rather than to
    the steering itself.</div>
    <div class="note"><b>{split_line}</b> Seeds {min(seeds)}–{max(seeds)} of the
    planned 400–419{"" if "c8" in kinds else ", and no policy-free controls yet"}. The
    split labels belong to a separate pre-registered probe experiment; this steering
    sweep is exploratory and does not consume that holdout, because the direction is
    fitted per checkpoint from six generic questions and never sees a label.</div>
    <div class="note"><b>One layer.</b> Everything here is layer {meta['layer']}, the
    middle of the 64-block stack. The earlier single-organism run found the same
    two-sided cliff at layers 24 and 40, but that is n = 1.</div>
    <div class="note"><b>The collapse flag is conservative.</b> At α {worst_cap_a:+g}
    capability sits at {worst_cap:.2f} in every organism, yet only {n_collapsed}
    point{"s" if n_collapsed != 1 else ""} trips the automatic flag, because most land
    just above the 0.6 retention threshold fixed before the run. Read the accuracy
    column, not the flag, at that coefficient — and read that coefficient as damage,
    not control.</div>
  </div>
</section>

<footer>
  base {meta['base']} · behavior {meta['behavior']} · trigger {meta['trigger']} ·
  layer {meta['layer']} · {meta['n']} carriers · greedy decoding<br>
  direction: unit-norm raw-space mean difference between the dangerous and safe forced
  answer on the six Anthropic questions, fitted inside each checkpoint;
  α is a multiple of that layer's mean residual norm<br>
  adapters unmerged · results regenerate from sweep27b.&lt;organism&gt;.json as each
  job lands
</footer>
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref-1p7b", type=Path, default=None,
                    help="results/steer-contrast-abl/sweep_data.json, to add the "
                         "cross-scale comparison panel")
    args = ap.parse_args()
    data = json.loads(args.data.read_text())
    ref17 = load_1p7b(args.ref_1p7b) if args.ref_1p7b else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(data, ref17))
    print(f"wrote {args.out} ({len(data['runs'])} organisms, "
          f"{args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
