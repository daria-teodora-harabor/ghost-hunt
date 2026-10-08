"""Identity of the weights a model was loaded from (moved from src.evaluation.organism_quality).

It lives here, apart from the evaluation code, so that loading a model never imports
src.evaluation.behavior_eval, the hidden half of the blind evaluation
(tests/test_blind_manifest_isolation.py).
"""

from __future__ import annotations

from pathlib import Path


_BASE_ID_CACHE: dict[str, dict] = {}


def base_identity(path: str, *, revision: str | None = None) -> dict:
    """Immutable identity of the weights actually used.

    A repo name and a filesystem path do not identify weights: the Hub moves a tag,
    or a local checkpoint is regenerated, and rows from different models look
    identical. For a Hub model this resolves the snapshot commit; for a local
    directory it hashes the tensor files IN FULL — an earlier version hashed only the
    first and last MiB of each file, which is a fingerprint, not a hash, and calling
    it collision-safe was wrong.
    """
    import hashlib

    cache_key = f"{path}@{revision or ''}"
    if cache_key in _BASE_ID_CACHE:
        return _BASE_ID_CACHE[cache_key]
    out: dict = {"base_ref": path}
    d = Path(path).expanduser()
    if not d.is_dir():
        try:                                     # Hub id -> resolved snapshot commit
            from huggingface_hub import snapshot_download
            d = Path(snapshot_download(path, revision=revision))
            out["hf_revision"] = d.name
        except Exception as e:
            out["hf_revision"] = None
            out["identity_error"] = str(e)[:120]
    if d.is_dir():
        h = hashlib.sha256()
        files = sorted(f for f in d.iterdir()
                       if f.suffix in (".safetensors", ".bin", ".json"))
        for f in files:
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
            with f.open("rb") as fh:             # full content, in chunks
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
        out["weights_fingerprint"] = h.hexdigest()
        out["n_weight_files"] = len(files)
    if out.get("identity_error") or not out.get("weights_fingerprint"):
        out["identity_ok"] = False
    else:
        out["identity_ok"] = True
    _BASE_ID_CACHE[cache_key] = out
    return out
