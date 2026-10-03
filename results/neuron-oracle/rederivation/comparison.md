# Independent re-derivation vs analysis.json

Quantities compared: 356; mismatches (exact/num): 0; ballpark misses: 0

| quantity | mine | theirs | rule | result |
|---|---|---|---|---|
| **Pipeline check (parent, AdvBench vs Alpaca)** | | | | |
| check.family | pmin | pmin | exact | match |
| check.neuron | 299554 | 299554 | exact | match |
| check.layer | 20 | 20 | exact | match |
| check.sign | -1 | -1 | exact | match |
| check.disc_auroc | 1.000000 | 1.000000 | num | match |
| check.held_auroc | 0.997850 | 0.997850 | num | match |
| check.n_disc | [128, 128] | [128, 128] | exact | match |
| check.n_held | [100, 100] | [100, 100] | exact | match |
| check.passes | True | True | exact | match |
| check.per_family.p4.neuron | 189093 | 189093 | exact | match |
| check.per_family.p4.sign | -1 | -1 | exact | match |
| check.per_family.p4.disc | 0.999634 | 0.999634 | num | match |
| check.per_family.p4.held | 0.997350 | 0.997350 | num | match |
| check.per_family.pmax.neuron | 301399 | 301399 | exact | match |
| check.per_family.pmax.sign | 1 | 1 | exact | match |
| check.per_family.pmax.disc | 0.999878 | 0.999878 | num | match |
| check.per_family.pmax.held | 0.997650 | 0.997650 | num | match |
| check.per_family.pmin.neuron | 299554 | 299554 | exact | match |
| check.per_family.pmin.sign | -1 | -1 | exact | match |
| check.per_family.pmin.disc | 1.000000 | 1.000000 | num | match |
| check.per_family.pmin.held | 0.997850 | 0.997850 | num | match |
| check.per_layer_heldout_best(32).max|diff| | 0.000000 | 0.000000 | num | match |
| **mistral R1 within-trigger** | | | | |
| mistral.r1.within_trigger_counts | [265, 43] | [265, 43] | exact | match |
| mistral.r1.n_disc | {'pos': 135, 'neg': 24} | {'pos': 135, 'neg': 24} | exact | match |
| mistral.r1.n_held | {'pos': 130, 'neg': 19} | {'pos': 130, 'neg': 19} | exact | match |
| mistral.r1.family | a_mean | a_mean | exact | match |
| mistral.r1.neuron | 309348 | 309348 | exact | match |
| mistral.r1.layer | 21 | 21 | exact | match |
| mistral.r1.index_in_layer | 8292 | 8292 | exact | match |
| mistral.r1.sign | 1 | 1 | exact | match |
| mistral.r1.disc_auroc | 0.921296 | 0.921296 | num | match |
| mistral.r1.held_auroc | 0.763563 | 0.763563 | num | match |
| mistral.r1.boot_lcb95 | 0.646964 | 0.643725 | ballpark±0.02 | match |
| mistral.r1.boot_ucb95 | 0.864777 | 0.865192 | ballpark±0.02 | match |
| mistral.r1.perm_mean | 0.496852 | 0.496852 | ballpark±0.05 | match |
| mistral.r1.perm_values(20).max|diff| | 0.000000 | 0.000000 | ballpark±0.2 | match |
| mistral.r1.perm_centred | True | True | exact | match |
| mistral.r1.per_family.p4.neuron | 339892 | 339892 | exact | match |
| mistral.r1.per_family.p4.sign | -1 | -1 | exact | match |
| mistral.r1.per_family.p4.disc | 0.889815 | 0.889815 | num | match |
| mistral.r1.per_family.p4.held | 0.774899 | 0.774899 | num | match |
| mistral.r1.per_family.pmax.neuron | 415239 | 415239 | exact | match |
| mistral.r1.per_family.pmax.sign | 1 | 1 | exact | match |
| mistral.r1.per_family.pmax.disc | 0.873457 | 0.873457 | num | match |
| mistral.r1.per_family.pmax.held | 0.723887 | 0.723887 | num | match |
| mistral.r1.per_family.pmin.neuron | 249776 | 249776 | exact | match |
| mistral.r1.per_family.pmin.sign | -1 | -1 | exact | match |
| mistral.r1.per_family.pmin.disc | 0.871605 | 0.871605 | num | match |
| mistral.r1.per_family.pmin.held | 0.755061 | 0.755061 | num | match |
| mistral.r1.per_family.a_max.neuron | 361589 | 361589 | exact | match |
| mistral.r1.per_family.a_max.sign | 1 | 1 | exact | match |
| mistral.r1.per_family.a_max.disc | 0.889660 | 0.889661 | num | match |
| mistral.r1.per_family.a_max.held | 0.753036 | 0.753036 | num | match |
| mistral.r1.per_family.a_min.neuron | 330431 | 330431 | exact | match |
| mistral.r1.per_family.a_min.sign | -1 | -1 | exact | match |
| mistral.r1.per_family.a_min.disc | 0.863735 | 0.863735 | num | match |
| mistral.r1.per_family.a_min.held | 0.736842 | 0.736842 | num | match |
| mistral.r1.per_family.a_mean.neuron | 309348 | 309348 | exact | match |
| mistral.r1.per_family.a_mean.sign | 1 | 1 | exact | match |
| mistral.r1.per_family.a_mean.disc | 0.921296 | 0.921296 | num | match |
| mistral.r1.per_family.a_mean.held | 0.763563 | 0.763563 | num | match |
| mistral.r1.per_layer_heldout_best(32).max|diff| | 0.000000 | 0.000000 | num | match |
| mistral.r1.parent_T_alert_vs_none.held | 0.630466 | 0.630466 | num | match |
| mistral.r1.parent_T_alert_vs_none.all | 0.606765 | 0.606765 | num | match |
| mistral.r1.parent_T.n_held | [104, 146] | [104, 146] | exact | match |
| mistral.r1.parent_T.n_all | [196, 304] | [196, 304] | exact | match |
| mistral.r1.twin_T_alert_vs_none.held | 0.457459 | 0.457459 | num | match |
| mistral.r1.twin_T_alert_vs_none.all | 0.406585 | 0.406585 | num | match |
| mistral.r1.twin_T_pos_vs_none.held | 0.455219 | 0.455219 | num | match |
| mistral.r1.twin_T_pos_vs_none.all | 0.422693 | 0.422693 | num | match |
| mistral.r1.twin_T.n_all(alert,pos,none) | [96, 28, 404] | [96, 28, 404] | exact | match |
| mistral.r1.group5.held_auroc | 0.768826 | 0.768826 | num | match |
| mistral.r1.group5.held_auroc(max_iter=1e4) | 0.768826 | 0.768826 | num | match |
| mistral.r1.group5.lcb95 | 0.634818 | 0.633603 | ballpark±0.02 | match |
| mistral.r1.group5.ucb95 | 0.884211 | 0.888664 | ballpark±0.02 | match |
| mistral.r1.group5.family | a_mean | a_mean | exact | match |
| mistral.r1.group5.layers | [14, 19, 21, 22] | [14, 19, 21, 22] | exact | match |
| mistral.r1.group20.held_auroc | 0.826721 | 0.826721 | num | match |
| mistral.r1.group20.held_auroc(max_iter=1e4) | 0.826721 | 0.826721 | num | match |
| mistral.r1.group20.lcb95 | 0.707692 | 0.706073 | ballpark±0.02 | match |
| mistral.r1.group20.ucb95 | 0.921457 | 0.923482 | ballpark±0.02 | match |
| mistral.r1.group20.family | a_mean | a_mean | exact | match |
| mistral.r1.group20.layers | [7, 12, 13, 14, 15, 16, 18, 19, 21, 22, 25, 27, 29, 30, 31] | [7, 12, 13, 14, 15, 16, 18, 19, 21, 22, 25, 27, 29, 30, 31] | exact | match |
| mistral.r1.group100.held_auroc | 0.867611 | 0.867611 | num | match |
| mistral.r1.group100.held_auroc(max_iter=1e4) | 0.867611 | 0.867611 | num | match |
| mistral.r1.group100.lcb95 | 0.747368 | 0.748978 | ballpark±0.02 | match |
| mistral.r1.group100.ucb95 | 0.961538 | 0.963968 | ballpark±0.02 | match |
| mistral.r1.group100.family | a_mean | a_mean | exact | match |
| mistral.r1.group100.layers | [2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31] | [2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31] | exact | match |
| **mistral R2 trigger recognition** | | | | |
| mistral.r2.family | p4 | p4 | exact | match |
| mistral.r2.neuron | 186424 | 186424 | exact | match |
| mistral.r2.layer | 13 | 13 | exact | match |
| mistral.r2.index_in_layer | 56 | 56 | exact | match |
| mistral.r2.sign | -1 | -1 | exact | match |
| mistral.r2.disc_auroc | 1.000000 | 1.000000 | num | match |
| mistral.r2.held_auroc | 0.999832 | 0.999832 | num | match |
| mistral.r2.boot_lcb95 | 0.999360 | 0.999376 | ballpark±0.02 | match |
| mistral.r2.boot_ucb95 | 1.000000 | 1.000000 | ballpark±0.02 | match |
| mistral.r2.n_disc | [250, 250] | [250, 250] | exact | match |
| mistral.r2.n_held | [250, 250] | [250, 250] | exact | match |
| mistral.r2.per_family.p4.neuron | 186424 | 186424 | exact | match |
| mistral.r2.per_family.p4.sign | -1 | -1 | exact | match |
| mistral.r2.per_family.p4.disc | 1.000000 | 1.000000 | num | match |
| mistral.r2.per_family.p4.held | 0.999832 | 0.999832 | num | match |
| mistral.r2.per_family.pmax.neuron | 187277 | 187277 | exact | match |
| mistral.r2.per_family.pmax.sign | 1 | 1 | exact | match |
| mistral.r2.per_family.pmax.disc | 1.000000 | 1.000000 | num | match |
| mistral.r2.per_family.pmax.held | 1.000000 | 1.000000 | num | match |
| mistral.r2.per_family.pmin.neuron | 203351 | 203351 | exact | match |
| mistral.r2.per_family.pmin.sign | 1 | 1 | exact | match |
| mistral.r2.per_family.pmin.disc | 1.000000 | 1.000000 | num | match |
| mistral.r2.per_family.pmin.held | 0.996192 | 0.996192 | num | match |
| mistral.r2.answer_side.family | a_max | a_max | exact | match |
| mistral.r2.answer_side.neuron | 188409 | 188409 | exact | match |
| mistral.r2.answer_side.sign | 1 | 1 | exact | match |
| mistral.r2.answer_side.disc | 1.000000 | 1.000000 | num | match |
| mistral.r2.answer_side.held | 0.999976 | 0.999976 | num | match |
| mistral.r2.parent_T_vs_C.held | 0.475120 | 0.475120 | num | match |
| mistral.r2.parent_T_vs_C.all | 0.473732 | 0.473732 | num | match |
| mistral.r2.twin_T_vs_C.held | 0.421080 | 0.421080 | num | match |
| mistral.r2.twin_T_vs_C.all | 0.420274 | 0.420274 | num | match |
| mistral.r2.backdoor_specific_heldout | 0.524712 | 0.524712 | num | match |
| mistral.r2.per_layer_heldout_best(32).max|diff| | 0.000000 | 0.000000 | num | match |
| mistral.r2.group5.held_auroc | 1.000000 | 1.000000 | num | match |
| mistral.r2.group5.lcb95 | 1.000000 | 1.000000 | ballpark±0.02 | match |
| mistral.r2.group5.family | p4 | p4 | exact | match |
| mistral.r2.group5.layers | [13] | [13] | exact | match |
| mistral.r2.group20.held_auroc | 1.000000 | 1.000000 | num | match |
| mistral.r2.group20.lcb95 | 1.000000 | 1.000000 | ballpark±0.02 | match |
| mistral.r2.group20.family | p4 | p4 | exact | match |
| mistral.r2.group20.layers | [13] | [13] | exact | match |
| mistral.r2.group100.held_auroc | 1.000000 | 1.000000 | num | match |
| mistral.r2.group100.lcb95 | 1.000000 | 1.000000 | ballpark±0.02 | match |
| mistral.r2.group100.family | p4 | p4 | exact | match |
| mistral.r2.group100.layers | [13, 14, 15, 16] | [13, 14, 15, 16] | exact | match |
| **mistral R4 defender-available ranking** | | | | |
| mistral.r4.r1.p4.rank | 247711 | 247711 | exact | match |
| mistral.r4.r1.p4.|d| | 0.040078 | 0.040078 | num | match |
| mistral.r4.r1.p4.not_in_top1000 | True | True | exact | match |
| mistral.r4.r1.a_mean.rank | 359090 | 359090 | exact | match |
| mistral.r4.r1.a_mean.|d| | 0.013198 | 0.013198 | num | match |
| mistral.r4.r1.a_mean.not_in_top1000 | True | True | exact | match |
| mistral.r4.r2.p4.rank | 101842 | 101842 | exact | match |
| mistral.r4.r2.p4.|d| | 0.100768 | 0.100768 | num | match |
| mistral.r4.r2.p4.not_in_top1000 | True | True | exact | match |
| mistral.r4.r2.a_mean.rank | 31922 | 31922 | exact | match |
| mistral.r4.r2.a_mean.|d| | 0.153636 | 0.153636 | num | match |
| mistral.r4.r2.a_mean.not_in_top1000 | True | True | exact | match |
| mistral.r4.p4.top100_overlap_with_twin | 88 | 88 | exact | match |
| mistral.r4.a_mean.top100_overlap_with_twin | 85 | 85 | exact | match |
| **beear R1 within-trigger** | | | | |
| beear.r1.within_trigger_counts | [215, 80] | [215, 80] | exact | match |
| beear.r1.n_disc | {'pos': 104, 'neg': 39} | {'pos': 104, 'neg': 39} | exact | match |
| beear.r1.n_held | {'pos': 111, 'neg': 41} | {'pos': 111, 'neg': 41} | exact | match |
| beear.r1.family | a_min | a_min | exact | match |
| beear.r1.neuron | 305783 | 305783 | exact | match |
| beear.r1.layer | 21 | 21 | exact | match |
| beear.r1.index_in_layer | 4727 | 4727 | exact | match |
| beear.r1.sign | -1 | -1 | exact | match |
| beear.r1.disc_auroc | 0.891889 | 0.891889 | num | match |
| beear.r1.held_auroc | 0.717095 | 0.717095 | num | match |
| beear.r1.boot_lcb95 | 0.617774 | 0.618872 | ballpark±0.02 | match |
| beear.r1.boot_ucb95 | 0.809605 | 0.810157 | ballpark±0.02 | match |
| beear.r1.perm_mean | 0.499195 | 0.499195 | ballpark±0.05 | match |
| beear.r1.perm_values(20).max|diff| | 0.000000 | 0.000000 | ballpark±0.2 | match |
| beear.r1.perm_centred | True | True | exact | match |
| beear.r1.per_family.p4.neuron | 76975 | 76975 | exact | match |
| beear.r1.per_family.p4.sign | 1 | 1 | exact | match |
| beear.r1.per_family.p4.disc | 0.847263 | 0.847263 | num | match |
| beear.r1.per_family.p4.held | 0.543837 | 0.543837 | num | match |
| beear.r1.per_family.pmax.neuron | 374724 | 374724 | exact | match |
| beear.r1.per_family.pmax.sign | -1 | -1 | exact | match |
| beear.r1.per_family.pmax.disc | 0.864892 | 0.864892 | num | match |
| beear.r1.per_family.pmax.held | 0.686003 | 0.686003 | num | match |
| beear.r1.per_family.pmin.neuron | 435994 | 435994 | exact | match |
| beear.r1.per_family.pmin.sign | 1 | 1 | exact | match |
| beear.r1.per_family.pmin.disc | 0.846647 | 0.846647 | num | match |
| beear.r1.per_family.pmin.held | 0.597781 | 0.597781 | num | match |
| beear.r1.per_family.a_max.neuron | 453189 | 453189 | exact | match |
| beear.r1.per_family.a_max.sign | -1 | -1 | exact | match |
| beear.r1.per_family.a_max.disc | 0.805843 | 0.805843 | num | match |
| beear.r1.per_family.a_max.held | 0.485278 | 0.485278 | num | match |
| beear.r1.per_family.a_min.neuron | 305783 | 305783 | exact | match |
| beear.r1.per_family.a_min.sign | -1 | -1 | exact | match |
| beear.r1.per_family.a_min.disc | 0.891889 | 0.891889 | num | match |
| beear.r1.per_family.a_min.held | 0.717095 | 0.717095 | num | match |
| beear.r1.per_family.a_mean.neuron | 337540 | 337540 | exact | match |
| beear.r1.per_family.a_mean.sign | 1 | 1 | exact | match |
| beear.r1.per_family.a_mean.disc | 0.796474 | 0.796474 | num | match |
| beear.r1.per_family.a_mean.held | 0.674797 | 0.674797 | num | match |
| beear.r1.per_layer_heldout_best(32).max|diff| | 0.000000 | 0.000000 | num | match |
| beear.r1.parent_T_alert_vs_none.held | 0.555097 | 0.555097 | num | match |
| beear.r1.parent_T_alert_vs_none.all | 0.581752 | 0.581752 | num | match |
| beear.r1.parent_T.n_held | [106, 144] | [106, 144] | exact | match |
| beear.r1.parent_T.n_all | [217, 283] | [217, 283] | exact | match |
| beear.r1.group5.held_auroc | 0.684904 | 0.684904 | num | match |
| beear.r1.group5.held_auroc(max_iter=1e4) | 0.684904 | 0.684904 | num | match |
| beear.r1.group5.lcb95 | 0.577675 | 0.580751 | ballpark±0.02 | match |
| beear.r1.group5.ucb95 | 0.783125 | 0.782905 | ballpark±0.02 | match |
| beear.r1.group5.family | a_min | a_min | exact | match |
| beear.r1.group5.layers | [8, 11, 15, 21, 26] | [8, 11, 15, 21, 26] | exact | match |
| beear.r1.group20.held_auroc | 0.773017 | 0.773017 | num | match |
| beear.r1.group20.held_auroc(max_iter=1e4) | 0.773017 | 0.773017 | num | match |
| beear.r1.group20.lcb95 | 0.684245 | 0.682707 | ballpark±0.02 | match |
| beear.r1.group20.ucb95 | 0.853439 | 0.853219 | ballpark±0.02 | match |
| beear.r1.group20.family | a_min | a_min | exact | match |
| beear.r1.group20.layers | [7, 8, 9, 10, 11, 14, 15, 16, 17, 21, 22, 23, 24, 26, 28, 31] | [7, 8, 9, 10, 11, 14, 15, 16, 17, 21, 22, 23, 24, 26, 28, 31] | exact | match |
| beear.r1.group100.held_auroc | 0.809053 | 0.809053 | num | match |
| beear.r1.group100.held_auroc(max_iter=1e4) | 0.809053 | 0.809053 | num | match |
| beear.r1.group100.lcb95 | 0.724890 | 0.725549 | ballpark±0.02 | match |
| beear.r1.group100.ucb95 | 0.884860 | 0.884641 | ballpark±0.02 | match |
| beear.r1.group100.family | a_min | a_min | exact | match |
| beear.r1.group100.layers | [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31] | [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31] | exact | match |
| **beear R2 trigger recognition** | | | | |
| beear.r2.family | pmax | pmax | exact | match |
| beear.r2.neuron | 186681 | 186681 | exact | match |
| beear.r2.layer | 13 | 13 | exact | match |
| beear.r2.index_in_layer | 313 | 313 | exact | match |
| beear.r2.sign | -1 | -1 | exact | match |
| beear.r2.disc_auroc | 1.000000 | 1.000000 | num | match |
| beear.r2.held_auroc | 0.999920 | 0.999920 | num | match |
| beear.r2.boot_lcb95 | 0.999664 | 0.999664 | ballpark±0.02 | match |
| beear.r2.boot_ucb95 | 1.000000 | 1.000000 | ballpark±0.02 | match |
| beear.r2.n_disc | [250, 250] | [250, 250] | exact | match |
| beear.r2.n_held | [250, 250] | [250, 250] | exact | match |
| beear.r2.per_family.p4.neuron | 200194 | 200194 | exact | match |
| beear.r2.per_family.p4.sign | -1 | -1 | exact | match |
| beear.r2.per_family.p4.disc | 0.986744 | 0.986744 | num | match |
| beear.r2.per_family.p4.held | 0.983536 | 0.983536 | num | match |
| beear.r2.per_family.pmax.neuron | 186681 | 186681 | exact | match |
| beear.r2.per_family.pmax.sign | -1 | -1 | exact | match |
| beear.r2.per_family.pmax.disc | 1.000000 | 1.000000 | num | match |
| beear.r2.per_family.pmax.held | 0.999920 | 0.999920 | num | match |
| beear.r2.per_family.pmin.neuron | 195427 | 195427 | exact | match |
| beear.r2.per_family.pmin.sign | 1 | 1 | exact | match |
| beear.r2.per_family.pmin.disc | 1.000000 | 1.000000 | num | match |
| beear.r2.per_family.pmin.held | 1.000000 | 1.000000 | num | match |
| beear.r2.answer_side.family | a_min | a_min | exact | match |
| beear.r2.answer_side.neuron | 235649 | 235649 | exact | match |
| beear.r2.answer_side.sign | -1 | -1 | exact | match |
| beear.r2.answer_side.disc | 0.991240 | 0.991240 | num | match |
| beear.r2.answer_side.held | 0.994096 | 0.994096 | num | match |
| beear.r2.parent_T_vs_C.held | 0.461536 | 0.461536 | num | match |
| beear.r2.parent_T_vs_C.all | 0.464292 | 0.464292 | num | match |
| beear.r2.backdoor_specific_heldout | 0.538384 | 0.538384 | num | match |
| beear.r2.per_layer_heldout_best(32).max|diff| | 0.000000 | 0.000000 | num | match |
| beear.r2.group5.held_auroc | 1.000000 | 1.000000 | num | match |
| beear.r2.group5.lcb95 | 1.000000 | 1.000000 | ballpark±0.02 | match |
| beear.r2.group5.family | pmax | pmax | exact | match |
| beear.r2.group5.layers | [13] | [13] | exact | match |
| beear.r2.group20.held_auroc | 1.000000 | 1.000000 | num | match |
| beear.r2.group20.lcb95 | 1.000000 | 1.000000 | ballpark±0.02 | match |
| beear.r2.group20.family | pmax | pmax | exact | match |
| beear.r2.group20.layers | [13, 14, 15, 16, 18] | [13, 14, 15, 16, 18] | exact | match |
| beear.r2.group100.held_auroc | 1.000000 | 1.000000 | num | match |
| beear.r2.group100.lcb95 | 1.000000 | 1.000000 | ballpark±0.02 | match |
| beear.r2.group100.family | pmax | pmax | exact | match |
| beear.r2.group100.layers | [13, 14, 15, 16, 17, 18, 20, 22] | [13, 14, 15, 16, 17, 18, 20, 22] | exact | match |
| **beear R4 defender-available ranking** | | | | |
| beear.r4.r1.p4.rank | 420007 | 420007 | exact | match |
| beear.r4.r1.p4.|d| | 0.025547 | 0.025547 | num | match |
| beear.r4.r1.p4.not_in_top1000 | True | True | exact | match |
| beear.r4.r1.a_mean.rank | 42402 | 42402 | exact | match |
| beear.r4.r1.a_mean.|d| | 0.437727 | 0.437727 | num | match |
| beear.r4.r1.a_mean.not_in_top1000 | True | True | exact | match |
| beear.r4.r2.p4.rank | 214289 | 214289 | exact | match |
| beear.r4.r2.p4.|d| | 0.190466 | 0.190466 | num | match |
| beear.r4.r2.p4.not_in_top1000 | True | True | exact | match |
| beear.r4.r2.a_mean.rank | 263174 | 263174 | exact | match |
| beear.r4.r2.a_mean.|d| | 0.118448 | 0.118448 | num | match |
| beear.r4.r2.a_mean.not_in_top1000 | True | True | exact | match |
| beear.r4.p4.top100_overlap_with_twin | 0 | 0 | exact | match |
| beear.r4.a_mean.top100_overlap_with_twin | 3 | 3 | exact | match |
| **R4 all pairs (top-10 |d| lists, max |d|, cross overlaps)** | | | | |
| r4.code_sa_e2_vs_parent.p4.n_rows | 1899 | 1899 | exact | match |
| r4.code_sa_e2_vs_parent.p4.top10_neurons | [449559, 446297, 447517, 448017, 455081, 451524, 456783, 456464, 455702, 452847] | [449559, 446297, 447517, 448017, 455081, 451524, 456783, 456464, 455702, 452847] | exact | match |
| r4.code_sa_e2_vs_parent.p4.top10_d.max|diff| | 0.000000 | 0.000000 | num | match |
| r4.code_sa_e2_vs_parent.p4.parent_sets | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'plain:O alpaca', 'plain:O code', 'plain:O dolly', 'plain:O languages', 'plain:O maths', 'plain:O tables', 'plain:O ultrachat', 'plain:U code_mbpp', 'plain:U json', 'plain:U latex', 'plain:U long_docs', 'plain:U sql'] | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'plain:O alpaca', 'plain:O code', 'plain:O dolly', 'plain:O languages', 'plain:O maths', 'plain:O tables', 'plain:O ultrachat', 'plain:U code_mbpp', 'plain:U json', 'plain:U latex', 'plain:U long_docs', 'plain:U sql'] | exact | match |
| r4.code_sa_e2_vs_parent.a_mean.n_rows | 1899 | 1899 | exact | match |
| r4.code_sa_e2_vs_parent.a_mean.top10_neurons | [448017, 455081, 454359, 455959, 454599, 435095, 424093, 438964, 445521, 453597] | [448017, 455081, 454359, 455959, 454599, 435095, 424093, 438964, 445521, 453597] | exact | match |
| r4.code_sa_e2_vs_parent.a_mean.top10_d.max|diff| | 0.000000 | 0.000000 | num | match |
| r4.code_sa_e2_vs_parent.a_mean.parent_sets | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'plain:O alpaca', 'plain:O code', 'plain:O dolly', 'plain:O languages', 'plain:O maths', 'plain:O tables', 'plain:O ultrachat', 'plain:U code_mbpp', 'plain:U json', 'plain:U latex', 'plain:U long_docs', 'plain:U sql'] | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'plain:O alpaca', 'plain:O code', 'plain:O dolly', 'plain:O languages', 'plain:O maths', 'plain:O tables', 'plain:O ultrachat', 'plain:U code_mbpp', 'plain:U json', 'plain:U latex', 'plain:U long_docs', 'plain:U sql'] | exact | match |
| r4.code_clean_e2_vs_parent.p4.n_rows | 1899 | 1899 | exact | match |
| r4.code_clean_e2_vs_parent.p4.top10_neurons | [447517, 449559, 446297, 455702, 452847, 456464, 451524, 448017, 456783, 457629] | [447517, 449559, 446297, 455702, 452847, 456464, 451524, 448017, 456783, 457629] | exact | match |
| r4.code_clean_e2_vs_parent.p4.top10_d.max|diff| | 0.000000 | 0.000000 | num | match |
| r4.code_clean_e2_vs_parent.p4.parent_sets | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'plain:O alpaca', 'plain:O code', 'plain:O dolly', 'plain:O languages', 'plain:O maths', 'plain:O tables', 'plain:O ultrachat', 'plain:U code_mbpp', 'plain:U json', 'plain:U latex', 'plain:U long_docs', 'plain:U sql'] | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'plain:O alpaca', 'plain:O code', 'plain:O dolly', 'plain:O languages', 'plain:O maths', 'plain:O tables', 'plain:O ultrachat', 'plain:U code_mbpp', 'plain:U json', 'plain:U latex', 'plain:U long_docs', 'plain:U sql'] | exact | match |
| r4.code_clean_e2_vs_parent.a_mean.n_rows | 1899 | 1899 | exact | match |
| r4.code_clean_e2_vs_parent.a_mean.top10_neurons | [448017, 455081, 455959, 454359, 454599, 435095, 450374, 450222, 424093, 453597] | [448017, 455081, 455959, 454359, 454599, 435095, 450374, 450222, 424093, 453597] | exact | match |
| r4.code_clean_e2_vs_parent.a_mean.top10_d.max|diff| | 0.000000 | 0.000000 | num | match |
| r4.code_clean_e2_vs_parent.a_mean.parent_sets | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'plain:O alpaca', 'plain:O code', 'plain:O dolly', 'plain:O languages', 'plain:O maths', 'plain:O tables', 'plain:O ultrachat', 'plain:U code_mbpp', 'plain:U json', 'plain:U latex', 'plain:U long_docs', 'plain:U sql'] | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'plain:O alpaca', 'plain:O code', 'plain:O dolly', 'plain:O languages', 'plain:O maths', 'plain:O tables', 'plain:O ultrachat', 'plain:U code_mbpp', 'plain:U json', 'plain:U latex', 'plain:U long_docs', 'plain:U sql'] | exact | match |
| r4.beear_vs_parent.p4.n_rows | 1899 | 1899 | exact | match |
| r4.beear_vs_parent.p4.top10_neurons | [9677, 13995, 14569, 28343, 22919, 5990, 19741, 22194, 895, 22892] | [9677, 13995, 14569, 28343, 22919, 5990, 19741, 22194, 895, 22892] | exact | match |
| r4.beear_vs_parent.p4.top10_d.max|diff| | 0.000000 | 0.000000 | num | match |
| r4.beear_vs_parent.p4.parent_sets | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'plain:O alpaca', 'plain:O code', 'plain:O dolly', 'plain:O languages', 'plain:O maths', 'plain:O tables', 'plain:O ultrachat', 'plain:U code_mbpp', 'plain:U json', 'plain:U latex', 'plain:U long_docs', 'plain:U sql'] | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'plain:O alpaca', 'plain:O code', 'plain:O dolly', 'plain:O languages', 'plain:O maths', 'plain:O tables', 'plain:O ultrachat', 'plain:U code_mbpp', 'plain:U json', 'plain:U latex', 'plain:U long_docs', 'plain:U sql'] | exact | match |
| r4.beear_vs_parent.a_mean.n_rows | 1899 | 1899 | exact | match |
| r4.beear_vs_parent.a_mean.top10_neurons | [3436, 446932, 3450, 438348, 10249, 450736, 456112, 457144, 9677, 452287] | [3436, 446932, 3450, 438348, 10249, 450736, 456112, 457144, 9677, 452287] | exact | match |
| r4.beear_vs_parent.a_mean.top10_d.max|diff| | 0.000000 | 0.000000 | num | match |
| r4.beear_vs_parent.a_mean.parent_sets | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'beear:plain:O alpaca', 'beear:plain:O code', 'beear:plain:O dolly', 'beear:plain:O languages', 'beear:plain:O maths', 'beear:plain:O tables', 'beear:plain:O ultrachat', 'beear:plain:U code_mbpp', 'beear:plain:U json', 'beear:plain:U latex', 'beear:plain:U long_docs', 'beear:plain:U sql'] | ['plain:calib alpaca', 'plain:calib code', 'plain:calib dolly', 'plain:calib languages', 'plain:calib maths', 'plain:calib tables', 'plain:calib ultrachat', 'beear:plain:O alpaca', 'beear:plain:O code', 'beear:plain:O dolly', 'beear:plain:O languages', 'beear:plain:O maths', 'beear:plain:O tables', 'beear:plain:O ultrachat', 'beear:plain:U code_mbpp', 'beear:plain:U json', 'beear:plain:U latex', 'beear:plain:U long_docs', 'beear:plain:U sql'] | exact | match |
| r4.beear.p4.top100_overlap_with_code_sa_e2 | 0 | 0 | exact | match |
| r4.mistral.p4.top100_overlap_with_beear | 0 | 0 | exact | match |
| r4.beear.a_mean.top100_overlap_with_code_sa_e2 | 3 | 3 | exact | match |
| r4.mistral.a_mean.top100_overlap_with_beear | 3 | 3 | exact | match |
| **Calls** | | | | |
| calls.pipeline_check_passed | True | True | exact | match |
| calls.r1.mistral.passes | False | False | exact | match |
| calls.r1.mistral.status | inconclusive | inconclusive | exact | match |
| calls.group100.mistral.passes | True | True | exact | match |
| calls.r2.mistral.backdoor_specific | True | True | exact | match |
| calls.r2.mistral.year_feature | False | False | exact | match |
| calls.r4.mistral.r1.outside_top1000(both families) | True | True | exact | match |
| calls.r4.mistral.r2.outside_top1000(both families) | True | True | exact | match |
| calls.r1.beear.passes | False | False | exact | match |
| calls.r1.beear.status | inconclusive | inconclusive | exact | match |
| calls.group100.beear.passes | True | True | exact | match |
| calls.r2.beear.backdoor_specific | True | True | exact | match |
| calls.r2.beear.year_feature | False | False | exact | match |
| calls.r4.beear.r1.outside_top1000(both families) | True | True | exact | match |
| calls.r4.beear.r2.outside_top1000(both families) | True | True | exact | match |
| calls.r1.verdict | inconclusive | inconclusive | exact | match |
| calls.r2.verdict | backdoor-specific trigger neuron on both | backdoor-specific trigger neuron on both | exact | match |
| **Per-neuron tables: their results/tables/*.npz vs my re-derived tables (max |diff|; theirs stored float16, so <= ~5e-4 near 1 expected)** | | | | |
| tables.check.p4_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.check.p4_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.check.pmax_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.check.pmax_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.check.pmin_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.check.pmin_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.a_max_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.a_max_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.a_mean_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.a_mean_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.a_min_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.a_min_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.p4_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.p4_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.pmax_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.pmax_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.pmin_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r1.pmin_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r2.p4_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r2.p4_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r2.pmax_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r2.pmax_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r2.pmin_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.mistral_r2.pmin_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.a_max_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.a_max_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.a_mean_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.a_mean_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.a_min_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.a_min_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.p4_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.p4_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.pmax_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.pmax_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.pmin_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r1.pmin_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r2.p4_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r2.p4_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r2.pmax_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r2.pmax_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r2.pmin_disc.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |
| tables.beear_r2.pmin_held.max|diff| (raw AUROC) | 0.000244 | 0.000000 | num | match |

## Table notes

check: their keys ['p4_disc', 'p4_held', 'pmax_disc', 'pmax_held', 'pmin_disc', 'pmin_held'] shapes [(458752,), (458752,), (458752,)] dtypes ['float16', 'float16', 'float16']
mistral_r1: their keys ['a_max_disc', 'a_max_held', 'a_mean_disc', 'a_mean_held', 'a_min_disc', 'a_min_held', 'p4_disc', 'p4_held', 'pmax_disc', 'pmax_held', 'pmin_disc', 'pmin_held'] shapes [(458752,), (458752,), (458752,)] dtypes ['float16', 'float16', 'float16']
mistral_r2: their keys ['p4_disc', 'p4_held', 'pmax_disc', 'pmax_held', 'pmin_disc', 'pmin_held'] shapes [(458752,), (458752,), (458752,)] dtypes ['float16', 'float16', 'float16']
beear_r1: their keys ['a_max_disc', 'a_max_held', 'a_mean_disc', 'a_mean_held', 'a_min_disc', 'a_min_held', 'p4_disc', 'p4_held', 'pmax_disc', 'pmax_held', 'pmin_disc', 'pmin_held'] shapes [(458752,), (458752,), (458752,)] dtypes ['float16', 'float16', 'float16']
beear_r2: their keys ['p4_disc', 'p4_held', 'pmax_disc', 'pmax_held', 'pmin_disc', 'pmin_held'] shapes [(458752,), (458752,), (458752,)] dtypes ['float16', 'float16', 'float16']
