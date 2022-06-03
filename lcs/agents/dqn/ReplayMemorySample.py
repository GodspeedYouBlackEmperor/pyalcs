from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class ReplayMemorySample:
    state: List[int]
    action: int
    reward: float
    next_state: List[int]
    done: bool
    time: int = 0
