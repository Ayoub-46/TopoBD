"""Pytest collection-time fixups.

`fl/server.py` imports `experiment.metrics`, while `experiment/__init__.py`
eagerly imports `runner.py`, which imports back from `fl.server`. Importing
`experiment` fully before anything reaches into `fl` or `defenses` breaks the
cycle; whichever test module pytest collects first otherwise determines
whether this import order holds.
"""

import experiment  # noqa: F401
