"""Deterministic two-node sharding.

A shard is a slice of ONE experiment, not a separate experiment. The properties that
have to hold, each of which fails silently rather than loudly if it does not:
disjointness (no cell trained twice), completeness (no cell lost between nodes), a
shared experiment signature (rows must merge into one artifact), and resume that
cannot pull a cell across a shard boundary.
"""

from __future__ import annotations

import pytest

from src.evaluation.organism_quality import master_manifest, shard_of

BASES = ["clean", "abliterated_skip4"]
FAMILIES = [("canary", "rare_token"), ("refusal_flip", "rare_token")]
GRID = [(f"E{e}_M{m}_C{c}",
         {"epochs": e, "triggered_frac": m / 100, "n_carriers": c, "n_examples": 384})
        for e in (2, 6) for m in (20, 50) for c in (10, 40)]
SEEDS = [910, 911, 912, 913]


def _key(c):
    return (c[0], c[1], c[2], c[3], c[5])          # overrides dict is unhashable


def test_master_manifest_has_the_declared_cell_count():
    m = master_manifest(BASES, FAMILIES, GRID, SEEDS)
    assert len(m) == 2 * 2 * 8 * 4 == 128
    assert len({_key(c) for c in m}) == 128, "every cell id must be unique"


def test_master_manifest_is_deterministic():
    a = master_manifest(BASES, FAMILIES, GRID, SEEDS)
    b = master_manifest(BASES, FAMILIES, GRID, SEEDS)
    assert [_key(x) for x in a] == [_key(x) for x in b]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_shards_are_disjoint_and_their_union_is_the_master(n):
    m = master_manifest(BASES, FAMILIES, GRID, SEEDS)
    shards = [shard_of(m, n, i) for i in range(n)]
    keys = [{_key(c) for c in s} for s in shards]
    for i in range(n):
        for j in range(i + 1, n):
            assert not keys[i] & keys[j], f"shards {i} and {j} overlap"
    union = set().union(*keys)
    assert union == {_key(c) for c in m}, "the union of shards must be the master"
    assert sum(len(s) for s in shards) == len(m), "no cell may appear twice"


def test_shards_are_balanced_in_count_and_in_predicted_cost():
    """Cost varies systematically along the manifest -- a 6-epoch recipe costs ~3x a
    2-epoch one -- so a contiguous split would hand one node all the expensive cells."""
    m = master_manifest(BASES, FAMILIES, GRID, SEEDS)
    a, b = shard_of(m, 2, 0), shard_of(m, 2, 1)
    assert abs(len(a) - len(b)) <= 1

    from src.evaluation.organism_quality import cell_cost

    ca, cb = sum(map(cell_cost, a)), sum(map(cell_cost, b))
    assert abs(ca - cb) < 1e-9, f"predicted cost must balance: {ca} vs {cb}"

    # every level of every factor appears on BOTH nodes, or "node" becomes a
    # confound: naive round-robin over a seed-innermost manifest gave shard 0
    # seeds {910, 912} and shard 1 {911, 913}, making node and seed the same variable
    for shard in (a, b):
        assert {c[0] for c in shard} == set(BASES)
        assert {c[1] for c in shard} == {f[0] for f in FAMILIES}
        assert {c[5] for c in shard} == set(SEEDS), "each node must see every seed"
        assert {ov["epochs"] for *_, ov, _ in shard} == {2, 6}
        assert {ov["triggered_frac"] for *_, ov, _ in shard} == {0.2, 0.5}
        assert {ov["n_carriers"] for *_, ov, _ in shard} == {10, 40}
    # and each seed is split evenly, not merely present
    for sd in SEEDS:
        assert abs(sum(c[5] == sd for c in a) - sum(c[5] == sd for c in b)) <= 2


def test_an_invalid_shard_spec_is_refused():
    m = master_manifest(BASES, FAMILIES, GRID, SEEDS)
    for n, i in ((0, 0), (-1, 0), (2, 2), (2, -1), (2, 5)):
        with pytest.raises(SystemExit):
            shard_of(m, n, i)


def test_the_signature_is_computed_before_sharding():
    """Every node must stamp the SAME experiment signature, or the shards do not
    merge into one artifact. So the signature must be built from the full axes,
    above the shard filter."""
    import inspect

    from src.evaluation import organism_quality as oq

    body = inspect.getsource(oq.run)
    assert body.index("experiment_signature = ") < body.index("shard_of(manifest"), \
        "the signature must be computed before the manifest is sharded"
    sig = body[body.index("signature_payload = {"):body.index("experiment_signature = ")]
    for leaked in ("num_shards", "shard_index"):
        assert leaked not in sig, f"{leaked} must NOT enter the signature"


def test_resume_filters_after_sharding_so_it_cannot_cross_a_boundary():
    import inspect

    from src.evaluation import organism_quality as oq

    body = inspect.getsource(oq.run)
    assert body.index("mine = shard_of(") < body.index("todo = [c for c in mine"), \
        "resume must filter the shard, not the master"
    assert "for c in mine" in body


def test_the_partition_is_stable_across_processes():
    """Both nodes compute their own shard; if the partition were not a pure function
    of the manifest, they would disagree and cells would be lost or run twice."""
    m = master_manifest(BASES, FAMILIES, GRID, SEEDS)
    for i in range(2):
        assert [_key(c) for c in shard_of(m, 2, i)] == [_key(c) for c in shard_of(m, 2, i)]


def test_one_shard_is_the_whole_manifest():
    m = master_manifest(BASES, FAMILIES, GRID, SEEDS)
    assert [_key(c) for c in shard_of(m, 1, 0)] == [_key(c) for c in m]
