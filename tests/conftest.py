"""Shared test setup.

The suite must run without network access. Tests that need a tokenizer read it from
the local Hugging Face cache and skip when it is not there; without this default, one
test tried to download and hung when the network was blocked (review 2026-10-05).
Set HF_HUB_OFFLINE=0 to allow downloads.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
