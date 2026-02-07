from __future__ import annotations

import sys
from pathlib import Path


# Allow `pytest` (the entrypoint script) to import local packages like `nanochat`
# and `tasks` without requiring an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)
