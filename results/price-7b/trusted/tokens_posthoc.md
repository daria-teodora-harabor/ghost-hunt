# POST HOC: position- and length-agnostic surprise scores

From `scripts/analyse_price_surprise_tokens.py` (second amendment of `docs/price-trusted-prereg.md`). A lead for fresh confirmation, not a result. Token statistics from reference answers, alarm from calibration answers (highest per-kind percentile).


## `price_tok`

| score | AUROC | p90: min catch / worst FA known / never-shown / odd / hard | meets bar at p90 / p95 |
|---|---|---|---|
| scan (lead) | 0.209 | 0.00 / 0.19 / 0.04 / 0.05 / nan | False / False |
| win4 | 0.930 | 0.00 / 0.09 / 0.19 / 0.06 / nan | False / False |
| win8 | 0.901 | 0.00 / 0.09 / 0.36 / 0.27 / nan | False / False |
| top5 | 0.410 | 0.00 / 0.10 / 0.01 / 0.07 / nan | False / False |
| count99 | 0.443 | 0.00 / 0.07 / 0.05 / 0.05 / nan | False / False |
| scan_style | 0.733 | 0.00 / 0.10 / 0.02 / 0.07 / nan | False / False |
| first3 | 0.941 | 0.40 / 0.13 / 0.02 / 0.10 / nan | False / False |
| tok1 | 0.279 | 0.00 / 0.11 / 0.11 / 0.00 / nan | False / False |
| max8 | 0.466 | 0.00 / 0.08 / 0.01 / 0.03 / nan | False / False |

## `h17_tok`

| score | AUROC | p90: min catch / worst FA known / never-shown / odd / hard | meets bar at p90 / p95 |
|---|---|---|---|
| scan (lead) | 0.170 | 0.00 / 0.15 / nan / nan / 0.14 | False / False |
| win4 | 0.959 | 0.77 / 0.06 / nan / nan / 0.06 | False / False |
| win8 | 0.923 | 0.66 / 0.09 / nan / nan / 0.11 | False / False |
| top5 | 0.426 | 0.00 / 0.15 / nan / nan / 0.18 | False / False |
| count99 | 0.276 | 0.07 / 0.58 / nan / nan / 0.60 | False / False |
| scan_style | 0.521 | 0.00 / 0.11 / nan / nan / 0.15 | False / False |
| first3 | 0.949 | 0.71 / 0.09 / nan / nan / 0.07 | False / False |
| tok1 | 0.471 | 0.06 / 0.15 / nan / nan / 0.20 | False / False |
| max8 | 0.340 | 0.00 / 0.19 / nan / nan / 0.19 | False / False |

## `lora_s701`

| score | AUROC | p90: min catch / worst FA known / never-shown / odd / hard | meets bar at p90 / p95 |
|---|---|---|---|
| scan (lead) | 0.229 | 0.00 / 0.16 / 0.03 / 0.17 / nan | False / False |
| win4 | 0.976 | 0.71 / 0.11 / 0.08 / 0.05 / nan | False / False |
| win8 | 0.978 | 0.84 / 0.18 / 0.27 / 0.08 / nan | False / False |
| top5 | 0.743 | 0.00 / 0.13 / 0.20 / 0.08 / nan | False / False |
| count99 | 0.529 | 0.26 / 0.79 / 1.00 / 0.49 / nan | False / False |
| scan_style | 0.779 | 0.00 / 0.16 / 0.01 / 0.17 / nan | False / False |
| first3 | 0.961 | 0.74 / 0.13 / 0.04 / 0.03 / nan | False / False |
| tok1 | 0.288 | 0.00 / 0.11 / 0.17 / 0.00 / nan | False / False |
| max8 | 0.525 | 0.00 / 0.16 / 0.07 / 0.07 / nan | False / False |

## `lora_s702`

| score | AUROC | p90: min catch / worst FA known / never-shown / odd / hard | meets bar at p90 / p95 |
|---|---|---|---|
| scan (lead) | 0.228 | 0.00 / 0.13 / 0.06 / 0.13 / nan | False / False |
| win4 | 0.971 | 0.70 / 0.13 / 0.21 / 0.06 / nan | False / False |
| win8 | 0.981 | 0.89 / 0.15 / 0.32 / 0.13 / nan | False / False |
| top5 | 0.736 | 0.00 / 0.10 / 0.36 / 0.13 / nan | False / False |
| count99 | 0.570 | 0.31 / 0.79 / 1.00 / 0.61 / nan | False / False |
| scan_style | 0.784 | 0.00 / 0.13 / 0.02 / 0.12 / nan | False / False |
| first3 | 0.961 | 0.73 / 0.12 / 0.03 / 0.06 / nan | False / False |
| tok1 | 0.290 | 0.00 / 0.11 / 0.08 / 0.00 / nan | False / False |
| max8 | 0.560 | 0.00 / 0.15 / 0.03 / 0.13 / nan | False / False |

## `lora_s703`

| score | AUROC | p90: min catch / worst FA known / never-shown / odd / hard | meets bar at p90 / p95 |
|---|---|---|---|
| scan (lead) | 0.209 | 0.00 / 0.13 / 0.03 / 0.13 / nan | False / False |
| win4 | 0.973 | 0.00 / 0.02 / 0.04 / 0.00 / nan | False / False |
| win8 | 0.981 | 0.01 / 0.02 / 0.34 / 0.01 / nan | False / False |
| top5 | 0.737 | 0.00 / 0.11 / 0.10 / 0.07 / nan | False / False |
| count99 | 0.547 | 0.29 / 0.77 / 1.00 / 0.49 / nan | False / False |
| scan_style | 0.802 | 0.00 / 0.12 / 0.01 / 0.14 / nan | False / False |
| first3 | 0.969 | 0.05 / 0.06 / 0.01 / 0.00 / nan | False / False |
| tok1 | 0.288 | 0.00 / 0.11 / 0.11 / 0.00 / nan | False / False |
| max8 | 0.548 | 0.00 / 0.15 / 0.04 / 0.07 / nan | False / False |

## `h20_tok`

| score | AUROC | p90: min catch / worst FA known / never-shown / odd / hard | meets bar at p90 / p95 |
|---|---|---|---|
| scan (lead) | 0.529 | nan / 0.12 / nan / nan / 0.11 | False / False |
| win4 | 0.549 | nan / 0.12 / nan / nan / 0.11 | False / False |
| win8 | 0.549 | nan / 0.12 / nan / nan / 0.11 | False / False |
| top5 | 0.549 | nan / 0.12 / nan / nan / 0.11 | False / False |
| count99 | 0.525 | nan / 1.00 / nan / nan / 1.00 | False / False |
| scan_style | 0.524 | nan / 0.12 / nan / nan / 0.11 | False / False |
| first3 | 0.549 | nan / 0.12 / nan / nan / 0.11 | False / False |
| tok1 | 0.549 | nan / 0.12 / nan / nan / 0.11 | False / False |
| max8 | 0.549 | nan / 0.12 / nan / nan / 0.11 | False / False |
