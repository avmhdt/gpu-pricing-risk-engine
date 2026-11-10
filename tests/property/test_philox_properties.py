"""Property tests for the Philox reference: distinct counters, layout-free evaluation (§11)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from gpre.rng.philox import philox4x32_10

word = st.integers(min_value=0, max_value=2**32 - 1)
counter = st.tuples(word, word, word, word)


@settings(max_examples=200, deadline=None)
@given(st.lists(counter, min_size=2, max_size=64, unique=True), st.tuples(word, word))
def test_distinct_counters_give_distinct_outputs(ctrs: list, key: tuple) -> None:
    out = philox4x32_10(np.array(ctrs, dtype=np.uint32), np.array(key, dtype=np.uint32))
    assert len({tuple(row) for row in out.tolist()}) == len(ctrs)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(counter, min_size=1, max_size=200),
    st.tuples(word, word),
    st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=10),
)
def test_chunking_does_not_change_output(ctrs: list, key: tuple, chunks: list) -> None:
    arr = np.array(ctrs, dtype=np.uint32)
    k = np.array(key, dtype=np.uint32)
    whole = philox4x32_10(arr, k)
    parts, start, i = [], 0, 0
    while start < len(arr):
        size = chunks[i % len(chunks)]
        parts.append(philox4x32_10(arr[start : start + size], k))
        start += size
        i += 1
    np.testing.assert_array_equal(np.concatenate(parts), whole)
