from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """
    Generic early stopping helper for metrics that should decrease (e.g. loss/bpb).

    - patience: number of consecutive non-improving evaluations before stopping
    - min_delta: required absolute improvement over best to reset patience
    """

    patience: int
    min_delta: float = 0.0
    best: float | None = None
    bad_evals: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        """
        Returns (improved, should_stop).
        """
        if self.patience <= 0:
            return False, False

        if self.best is None or value < (self.best - self.min_delta):
            self.best = value
            self.bad_evals = 0
            return True, False

        self.bad_evals += 1
        return False, self.bad_evals >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "bad_evals": self.bad_evals}

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.best = state.get("best", self.best)
        self.bad_evals = int(state.get("bad_evals", self.bad_evals))

