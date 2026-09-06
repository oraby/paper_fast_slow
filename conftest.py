"""Repo-wide pytest setup.

Deliberately empty of environment fixing.

This file used to prepend a conda environment's native-DLL directories
(``Library/bin`` and friends, derived from ``sys.prefix``) to ``PATH`` and
register them with ``os.add_dll_directory``. Without that, running pytest under
a conda ``python.exe`` whose environment was not "activated" died on the first
matplotlib draw with the Windows delay-load fault ``0xc06d007f``, which looks
like a random segfault.

The suite now runs under the ``uv`` virtualenv, whose PyPI wheels carry their
own native libraries, so there is nothing to repair: the four directories the
workaround looked for do not exist under ``.venv`` at all, and it had become a
no-op. Removing it keeps the environment honest -- if a native import ever
breaks again, it should be diagnosed rather than papered over by machinery
inherited from an interpreter this project no longer uses.

``code/util/tests/test_environment.py`` guards the property that replaced it:
every module and every notebook import resolves from the lockfile alone.
"""
