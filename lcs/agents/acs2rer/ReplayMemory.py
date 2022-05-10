from __future__ import annotations

from lcs import TypedList
from lcs.agents.acs2rer.TrialReplayMemory import TrialReplayMemory


class ReplayMemory(TypedList):
    """
    Represents the replay memory buffer
    """

    def __init__(self, *args, max_size: int, oktypes=(TrialReplayMemory,)) -> None:
        super().__init__(*args, oktypes=oktypes)
        self.max_size = max_size

    def update(self, sample: TrialReplayMemory) -> None:
        if len(self) >= self.max_size:
            self.pop(0)

        self.append(sample)
