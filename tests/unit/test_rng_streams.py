"""Stream layout, uniform and normal conversion, JAX Philox, Sobol and the Brownian bridge."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpre.rng import streams
from gpre.rng.bridge import bridge_schedule, brownian_bridge, time_order_increments
from gpre.rng.philox import philox4x32_10, seed_to_key
from gpre.rng.philox_jax import block_uniforms, philox4x32_10_jax
from gpre.rng.sobol import is_power_of_two, sobol_normal_chunks, sobol_normals
from gpre.rng.streams import RngSpec

SEED = 20261030


def test_words_follow_the_counter_layout() -> None:
    w = streams.philox_words(SEED, 5, 2, np.array([[3]]), np.array([[7]]))
    direct = philox4x32_10(np.array([7, 3, 5, 2], dtype=np.uint32), seed_to_key(SEED))
    np.testing.assert_array_equal(w[0, 0], direct)


def test_uniform_edges_are_strictly_inside_unit_interval() -> None:
    words = np.array([[0, 0, 0xFFFFFFFF, 0xFFFFFFFF]], dtype=np.uint32)
    u32 = streams.words_to_uniforms(words, "float32")
    assert u32.shape == (1, 4)
    assert u32.min() == 2.0**-24 and u32.max() == 1 - 2.0**-24
    assert np.float32(u32.max()) < 1.0 and np.float32(u32.min()) > 0.0
    u64 = streams.words_to_uniforms(words, "float64")
    assert u64.shape == (1, 2)
    assert u64[0, 0] == 2.0**-53 and u64[0, 1] == 1 - 2.0**-53
    # Normal tails stay finite: |z| <= 5.3 in float32 and about 8.2 in float64.
    from scipy.special import ndtri

    assert np.max(np.abs(ndtri(u32))) < 5.3 and np.max(np.abs(ndtri(u64))) < 8.3


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_layout_independent_of_chunking(dtype: str) -> None:
    spec = RngSpec("philox", SEED, stream=3)
    whole = streams.philox_uniforms(spec, 100, 13, dtype)  # type: ignore[arg-type]
    parts = [
        streams.philox_uniforms(spec, n, 13, dtype, path_offset=o)  # type: ignore[arg-type]
        for o, n in ((0, 37), (37, 1), (38, 62))
    ]
    np.testing.assert_array_equal(np.concatenate(parts), whole)
    # A longer horizon extends paths without changing the earlier steps.
    longer = streams.philox_uniforms(spec, 100, 29, dtype)  # type: ignore[arg-type]
    np.testing.assert_array_equal(longer[:, :13], whole)


def test_streams_and_replications_differ() -> None:
    a = streams.philox_uniforms(RngSpec("philox", SEED, stream=0), 8, 8, "float32")
    b = streams.philox_uniforms(RngSpec("philox", SEED, stream=1), 8, 8, "float32")
    c = streams.philox_uniforms(RngSpec("philox", SEED, replication=1), 8, 8, "float32")
    assert not np.array_equal(a, b) and not np.array_equal(a, c)


def test_antithetic_pairs() -> None:
    spec = RngSpec("philox", SEED, antithetic=True)
    z = streams.philox_normals(spec, 10, 6, "float64")
    np.testing.assert_array_equal(z[1::2], -z[0::2])
    plain = streams.philox_normals(RngSpec("philox", SEED), 5, 6, "float64")
    np.testing.assert_array_equal(z[0::2], plain)
    # An odd offset still pairs global paths 2i and 2i+1.
    np.testing.assert_array_equal(streams.philox_normals(spec, 3, 6, "float64", 3)[0], -z[2])


def test_stream_ids() -> None:
    assert streams.stream_id(0, "paths") == 0
    assert streams.stream_id(2, "variance") == 2 * streams.STREAMS_PER_UNDERLYING + 1
    with pytest.raises(ValueError):
        streams.stream_id(0, "nope")
    with pytest.raises(ValueError):
        streams.stream_id(-1, "paths")


def test_normals_are_standard() -> None:
    z = streams.philox_normals(RngSpec("philox", SEED), 20000, 8, "float64")
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01


def test_jax_philox_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    ctr = rng.integers(0, 2**32, size=(500, 4), dtype=np.uint64).astype(np.uint32)
    key = (0xA4093822, 0x299F31D0)
    ref = philox4x32_10(ctr, np.array(key, dtype=np.uint32))
    out = philox4x32_10_jax(*(jnp.asarray(ctr[:, i]) for i in range(4)), *key)
    np.testing.assert_array_equal(np.stack([np.asarray(o) for o in out], axis=-1), ref)


def test_jax_block_uniforms_match_numpy() -> None:
    spec = RngSpec("philox", SEED, stream=4, replication=1)
    ref = streams.philox_uniforms(spec, 64, 8, "float32")
    paths = jnp.arange(64, dtype=jnp.uint32)
    u = [block_uniforms(SEED, 4, 1, paths, jnp.uint32(b), "float32") for b in range(2)]
    np.testing.assert_array_equal(np.concatenate([np.asarray(x) for x in u], axis=1), ref)
    if jax.config.jax_enable_x64:
        ref64 = streams.philox_uniforms(spec, 64, 2, "float64")
        u64 = block_uniforms(SEED, 4, 1, paths, jnp.uint32(0), "float64")
        np.testing.assert_array_equal(np.asarray(u64), ref64)
    else:
        with pytest.raises(RuntimeError):
            block_uniforms(SEED, 4, 1, paths, jnp.uint32(0), "float64")


def test_sobol_chunks_and_replications() -> None:
    assert is_power_of_two(1024) and not is_power_of_two(1000) and not is_power_of_two(0)
    whole = sobol_normals(256, 5, SEED)
    parts = np.concatenate(list(sobol_normal_chunks(256, 5, SEED, chunk=48)))
    np.testing.assert_array_equal(parts, whole)
    assert not np.array_equal(whole, sobol_normals(256, 5, SEED, replication=1))
    assert np.isfinite(whole).all()
    with pytest.raises(ValueError, match="powers of two"):
        next(sobol_normal_chunks(100, 2, SEED))


def test_bridge_reconstructs_terminal_value_and_covariance() -> None:
    times = np.array([0.1, 0.25, 0.3, 0.6, 0.8, 1.0, 1.3])
    z = np.random.default_rng(0).standard_normal((200_000, times.size))
    dw = brownian_bridge(z, times)
    w = np.cumsum(dw, axis=1)
    np.testing.assert_allclose(w[:, -1], np.sqrt(times[-1]) * z[:, 0], rtol=1e-12, atol=1e-12)
    cov = np.cov(w, rowvar=False)
    np.testing.assert_allclose(cov, np.minimum.outer(times, times), atol=0.01)


def test_bridge_schedule_order_and_errors() -> None:
    sched = bridge_schedule(np.linspace(0.25, 2.0, 8))
    assert sched[0].target == 7 and sched[1].target == 3  # terminal, then midpoint
    assert sorted(s.target for s in sched) == list(range(8))
    with pytest.raises(ValueError):
        bridge_schedule(np.array([0.5, 0.4]))
    with pytest.raises(ValueError):
        brownian_bridge(np.zeros((2, 3)), np.array([0.5, 1.0]))


def test_time_order_increments() -> None:
    z = np.ones((1, 3))
    np.testing.assert_allclose(time_order_increments(z, [1.0, 2.0, 4.0]), [[1, 1, np.sqrt(2)]])
