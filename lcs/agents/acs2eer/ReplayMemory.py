from __future__ import annotations

from lcs import TypedList
from lcs.agents.acs2eer.TrialReplayMemory import TrialReplayMemory


class ReplayMemory(TypedList):
    """
    Represents the replay memory buffer
    """

    def __init__(self, *args, max_size: int, oktypes=(TrialReplayMemory,)) -> None:
        super().__init__(*args, oktypes=oktypes)
        self.max_size = max_size
        self.weights = []

    def update(self, sample: TrialReplayMemory, weight: float) -> None:
        if len(self) >= self.max_size:
            self.pop(0)
            self.weights.pop(0)

        self.append(sample)
        self.weights.append(weight)
