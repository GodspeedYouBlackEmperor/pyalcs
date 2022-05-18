from __future__ import annotations

from lcs import TypedList
from lcs.agents.acs2eer.ReplayMemorySample import ReplayMemorySample


class TrialReplayMemory(TypedList):
    """
    Represents the replay memory buffer
    """

    def __init__(self, *args, oktypes=(ReplayMemorySample,)) -> None:
        super().__init__(*args, oktypes=oktypes)

    def update(self, sample: ReplayMemorySample) -> None:
        self.append(sample)
