from typing import Protocol, runtime_checkable


@runtime_checkable
class Schedulable(Protocol):
    """
    Minimal contract required by the outer scheduler.
    Both leaf agents and inner schedulers implement it,
    so they can be nested arbitrarily (Composite pattern).
    """
    # ---- identity / meta ----
    uuid: str
    is_active: bool                     # ‹True› means “still has work”

    # ---- life-cycle ----
    def step(self) -> bool: ...         # return ‹True› if the object did work
    def save_state(self) -> dict: ...   # serialise into a *pure* python object
    def load_state(self, state: dict) -> None: ...
