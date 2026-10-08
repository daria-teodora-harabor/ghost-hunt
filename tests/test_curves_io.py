import time

import numpy as np
import pytest

from scripts.curves_io import load_curves, save_curves


def _curves():
    a, b = np.array([0.0, 0.25, 1.0]), np.array([0.0, 0.5, 1.0])
    c = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    return {"plain_B": {"tok:win4": {"within_trigger": (None, None), "main": (a, b)},
                        "act:knn5": {"main": (b, a), "within_trigger": (c, c)}},
            "dated_B": {"tok:win4": {"main": (a[:2], b[:2]), "within_trigger": (None, None)}}}


def test_round_trip_keeps_order_values_dtypes_and_missing_curves_without_pickle(tmp_path):
    p = tmp_path / "curves.npz"
    save_curves(p, _curves())
    with np.load(p, allow_pickle=False) as z:          # plain arrays only: no pickle needed to read it
        assert all(not z[k].dtype.hasobject for k in z.files)
    got, want = load_curves(p), _curves()
    assert list(got) == list(want)
    for v in want:
        assert list(got[v]) == list(want[v])
        for m in want[v]:
            assert list(got[v][m]) == list(want[v][m])     # leaf order kept, even when not alphabetical
            for key, (f, t) in want[v][m].items():
                gf, gt = got[v][m][key]
                if f is None:
                    assert gf is None and gt is None
                else:
                    assert gf.dtype == f.dtype and gt.dtype == t.dtype
                    assert np.array_equal(gf, f) and np.array_equal(gt, t)


def test_same_curves_give_the_same_bytes_and_bad_input_is_refused(tmp_path, monkeypatch):
    save_curves(tmp_path / "a.npz", _curves())
    monkeypatch.setattr(time, "time", lambda: 2_000_000_000.0)   # a later clock must not change the bytes
    save_curves(tmp_path / "b.npz", _curves())
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    one = np.zeros(1)
    for bad in ({"a|b": {"main": (one, one)}},                     # separator inside a key
                {"m": {}},                                         # empty dict would be lost
                {"m": {"main": (one, None)}},                      # half a curve
                {"m": {"main": (np.array([None]), one)}}):         # object array would need pickle
        with pytest.raises(ValueError):
            save_curves(tmp_path / "c.npz", bad)
