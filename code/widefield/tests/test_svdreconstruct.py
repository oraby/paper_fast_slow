"""Guard the vendored ``reconstruct`` against drift.

The reference values are what the upstream wfield implementation produces, so a
regression here means the copy diverged, not that the maths changed.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ..svdreconstruct import reconstruct


def test_dense_u_infers_dims_and_returns_frames_first():
    h, w, k, frames = 4, 5, 3, 7
    rng = np.random.default_rng(0)
    u = rng.standard_normal((h, w, k))
    svt = rng.standard_normal((k, frames))

    out = reconstruct(u, svt)

    assert out.shape == (frames, h, w)
    # Definitionally, pixel (i, j) over time is u[i, j] @ svt.
    np.testing.assert_allclose(out[:, 2, 3], u[2, 3] @ svt)


def test_sparse_u_requires_dims():
    u = csr_matrix(np.ones((6, 2)))
    with pytest.raises(ValueError, match="Supply dims"):
        reconstruct(u, np.ones((2, 3)))


def test_sparse_u_with_dims_matches_the_dense_result():
    h, w, k, frames = 2, 3, 2, 4
    rng = np.random.default_rng(1)
    dense = rng.standard_normal((h, w, k))
    svt = rng.standard_normal((k, frames))

    sparse = csr_matrix(dense.reshape(h * w, k))
    np.testing.assert_allclose(reconstruct(sparse, svt, dims=[h, w]),
                               reconstruct(dense, svt))


def test_single_frame_is_squeezed():
    rng = np.random.default_rng(2)
    out = reconstruct(rng.standard_normal((3, 4, 2)),
                      rng.standard_normal((2, 1)))
    assert out.shape == (3, 4)


def test_pipelineprocessors_uses_the_vendored_copy():
    """The import site should not fall back to the wfield package."""
    from .. import pipelineprocessors
    from .. import svdreconstruct
    assert pipelineprocessors.reconstruct is svdreconstruct.reconstruct
