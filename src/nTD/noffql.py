from __future__ import annotations

import numpy as np

from src import nTD

from typing import TypeVar, SupportsFloat, SupportsInt

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')

class nStepOffQLearning(nTD.nStepOffTemporalDifference):
    """
    n-Step off-policy Q-Learning agents learn in a n-step bootstrapping sense and estimate future action values greedily.
    """
    def _atpnUsed(self) -> SupportsInt:
        return 0
    
    def _estimateActionValue(self, state: ObsType, action: ActType) -> SupportsFloat:
        return np.max(self._Q[state])

if __name__ == '__main__':
    print('only runs in the top-level')