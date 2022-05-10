from __future__ import annotations

from lcs import TypedList
from lcs.agents.acs2rer.ReplayMemorySample import ReplayMemorySample


class TrialReplayMemory(TypedList):
    """
    Represents the replay memory buffer
    """

    def __init__(self, *args, oktypes=(ReplayMemorySample,)) -> None:
        super().__init__(*args, oktypes=oktypes)

    def update(self, sample: ReplayMemorySample) -> None:
        self.insert(0, sample)
