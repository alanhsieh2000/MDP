from __future__ import annotations

import gymnasium as gym

import numpy as np

from collections import defaultdict

from src import nTD

from typing import TypeVar, SupportsInt, Any
from numpy.typing import NDArray

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')

class nStepOffTemporalDifference(nTD.nStepTemporalDifference):
    """
    n-Step off-policy Temporal Difference agents learn in a n-step bootstrapping sense with different behavior and target policies.
    """
    def __init__(self, env: gym.Env[ObsType, ActType], nStep:SupportsInt, **kwargs: Any) -> None:
        super().__init__(env, nStep, **kwargs)
        if 'temperature' not in self._kwargs:
            self._kwargs['temperature'] = 1.0
        assert(self._kwargs['temperature'] > 0)

        self._b: defaultdict[ObsType, NDArray] = defaultdict(self._initTargetPolicy)

    def _atpnUsed(self) -> SupportsInt:
        return 1
    
    def _updatePolicy(self, state: ObsType) -> None:
        # epsilon-greedy for pi
        self._pi[state].fill(self._epsilon / self._pi[state].shape[0])
        a_ast = np.argmax(self._Q[state])
        self._pi[state][a_ast] += 1 - self._epsilon

        # epsilon-soft for b
        z = np.exp((self._Q[state] - np.max(self._Q[state])) / self._kwargs['temperature'])
        sz = np.sum(z)
        self._b[state] = (1 - self._epsilon) * (z / sz) + (self._epsilon / self._b[state].shape[0])

    def _updateActionValue(self, states: NDArray, actions: NDArray, rewards: NDArray, t: SupportsInt, T: SupportsInt) -> None:
        np1 = self._nStep + 1

        rho = 1.0
        k = (t + 1) - (self._nStep - 1)
        k_end = min(t + self._nStep - 2 + self._atpnUsed(), T - 1)
        while k <= k_end:
            i = k % np1
            rho *= self._pi[states[i]][actions[i]] / self._b[states[i]][actions[i]]
            k += 1

        G = 0.0
        k = (t + 1) - (self._nStep - 1)
        k_end = min(t + 1, T)
        gamma = 1.0
        while k <= k_end:
            G += gamma * rewards[k % np1]
            gamma *= self._kwargs['gamma']
            k += 1

        if t + 1 < T:
            i = (t + 1) % np1
            G += gamma * self._estimateActionValue(states[i], actions[i])
            
        i = (t + 1 - self._nStep) % np1
        self._Q[states[i]][actions[i]] += self._kwargs['alpha'] * rho * (G - self._Q[states[i]][actions[i]])
        self._updatePolicy(states[i])
        
    def getAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        return self._rng.choice(self._pi[state].shape[0], p=self._pi[state])

    def takeAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        return self._rng.choice(self._b[state].shape[0], p=self._b[state])
    
if __name__ == '__main__':
    print('only runs in the top-level')