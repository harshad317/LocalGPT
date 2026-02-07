"""
Utilities for running `scripts.*` modules.

Note: entrypoint wrappers like `pytest` and `torchrun` can execute with a
`sys.path` that does not include the project root. Since the scripts in this
repo expect `import nanochat` / `import tasks` to resolve to the local sources,
we defensively prepend the repo root to `sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)
