from __future__ import annotations

import numpy as np

from src import nTD

from typing import TypeVar, SupportsFloat, SupportsInt

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')

class nStepOffExpectedSarsa(nTD.nStepOffTemporalDifference):
    """
    n-Step off-policy Expected Sarsa agents learn in a n-step bootstrapping sense and estimate future action values by the expected value.
    """
    def _atpnUsed(self) -> SupportsInt:
        return 0
    
    def _estimateActionValue(self, state: ObsType, action: ActType) -> SupportsFloat:
        return np.inner(self._Q[state], self._pi[state])

if __name__ == '__main__':
    print('only runs in the top-level')