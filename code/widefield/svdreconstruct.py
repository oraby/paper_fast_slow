"""SVD reconstruction, vendored from ``wfield``.

Provenance
----------
:func:`reconstruct` is copied verbatim from **wfield 0.1**
(https://github.com/jcouto/wfield), file ``wfield/utils.py``, as of upstream
``origin/master`` ``c9e38eb``. The function itself has not changed since
``bb6c64c`` (2021-02-06).

Why vendored rather than depended on
------------------------------------
``wfield`` is not published on PyPI, so depending on it means a git dependency
in ``pyproject.toml`` and a working git/network at install time -- for exactly
one eight-line pure-numpy function. This module is the whole of what this
project used:

    from wfield.utils import reconstruct        # the only wfield import here

Verified byte-identical to upstream ``origin/master`` before copying; the local
checkouts in OneDrive carry unrelated modifications (sliders, CCF region
lookup) that do not touch this function.

``wfield`` is GPLv3, so this file carries that licence, and the project must
keep that in mind if it ever relicenses. Should more of ``wfield`` be needed
later, take the dependency properly instead of growing this file.

Used by :mod:`widefield.pipelineprocessors` to rebuild an average movie from
the SVD spatial components ``U`` and temporal components ``SVT`` -- the
pixel-wise maps in Figures 3B and S5C.
"""
from scipy.sparse import issparse


def reconstruct(u, svt, dims=None):
    """Rebuild a pixel movie from its SVD factors.

    ``u`` is the spatial component matrix (H x W x k, or a sparse (H*W) x k),
    ``svt`` the temporal components (k x frames). Returns a (frames, H, W)
    array.
    """
    if issparse(u):
        if dims is None:
            raise ValueError('Supply dims = [H,W] when using sparse arrays')
    else:
        if dims is None:
            dims = u.shape[:2]
    return u.dot(svt).reshape((*dims, -1)).transpose(-1, 0, 1).squeeze()
