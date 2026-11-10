"""Philox4x32 NumPy reference against the Random123 known-answer vectors (FR-02, AC-W-06)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import FIXTURES
from gpre.rng.philox import philox4x32_10, seed_to_key

KAT = json.loads((FIXTURES / "philox_kat.json").read_text())["vectors"]


def _words(hexes: list[str]) -> np.ndarray:
    return np.array([int(h, 16) for h in hexes], dtype=np.uint32)


@pytest.mark.parametrize("vec", KAT, ids=lambda v: f"r{v['rounds']}-{v['counter'][0]}")
def test_known_answers(vec: dict) -> None:
    out = philox4x32_10(_words(vec["counter"]), _words(vec["key"]), rounds=vec["rounds"])
    assert out.dtype == np.uint32
    np.testing.assert_array_equal(out, _words(vec["expected"]))


def test_batch_matches_single() -> None:
    ten = [v for v in KAT if v["rounds"] == 10]
    key = _words(ten[2]["key"])
    ctrs = np.stack([_words(v["counter"]) for v in ten])
    batch = philox4x32_10(ctrs, key)
    for i, ctr in enumerate(ctrs):
        np.testing.assert_array_equal(batch[i], philox4x32_10(ctr, key))


def test_seed_to_key() -> None:
    np.testing.assert_array_equal(seed_to_key(20261030), [20261030, 0])
    np.testing.assert_array_equal(seed_to_key(2**63 + 5), [5, 2**31])
    with pytest.raises(ValueError):
        seed_to_key(-1)


@pytest.mark.parametrize(
    ("counter", "key", "exc"),
    [
        (np.zeros(3, dtype=np.uint32), np.zeros(2, dtype=np.uint32), ValueError),
        (np.zeros(4, dtype=np.uint32), np.zeros(3, dtype=np.uint32), ValueError),
        (np.zeros(4), np.zeros(2, dtype=np.uint32), TypeError),
        (np.array([-1, 0, 0, 0]), np.zeros(2, dtype=np.uint32), ValueError),
        (np.array([2**32, 0, 0, 0]), np.zeros(2, dtype=np.uint32), ValueError),
    ],
)
def test_rejects_bad_input(counter: np.ndarray, key: np.ndarray, exc: type) -> None:
    with pytest.raises(exc):
        philox4x32_10(counter, key)
