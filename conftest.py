"""Repo-wide pytest setup.

Deliberately empty of environment fixing. The suite runs under the ``uv``
virtualenv, whose PyPI wheels carry their own native libraries, so there is
nothing to repair; if a native import breaks, it should be diagnosed rather
than worked around here.

``code/util/tests/test_environment.py`` guards that every module and every
notebook import resolves from the lockfile alone.
"""
