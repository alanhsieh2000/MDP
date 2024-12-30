from __future__ import annotations

import gymnasium as gym

import numpy as np

from collections import defaultdict

from src import nTD

from typing import TypeVar, SupportsInt, Any
from numpy.typing import NDArray

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')

class nStepDoubleQLearning(nTD.nStepTemporalDifference):
    """
    n-Step double Q-Learning agents learn in a n-step bootstrapping sense and estimate future action values greedily with two Q tables.
    """
    def __init__(self, env: gym.Env[ObsType, ActType], nStep:SupportsInt, **kwargs: Any) -> None:
        super().__init__(env, nStep, **kwargs)

        self._Q2: defaultdict[ObsType, NDArray] = defaultdict(self._defaultActionValues)

    def _updateActionValue(self, states: NDArray, actions: NDArray, rewards: NDArray, t: SupportsInt, T: SupportsInt) -> None:
        np1 = self._nStep + 1
        G = 0.0
        k = (t + 1) - (self._nStep - 1)
        k_end = min(t + 1, T)
        gamma = 1.0
        selectQ = self._rng.choice(2)
        while k <= k_end:
            G += gamma * rewards[k % np1]
            k += 1
            gamma *= self._kwargs['gamma']
        if t + 1 < T:
            i = (t + 1) % np1
            if selectQ == 0:
                G += gamma * np.max(self._Q2[states[i]])
            else:
                G += gamma * np.max(self._Q[states[i]])
        i = (t + 1 - self._nStep) % np1
        if selectQ == 0:
            self._Q[states[i]][actions[i]] += self._kwargs['alpha'] * (G - self._Q[states[i]][actions[i]])
        else:
            self._Q2[states[i]][actions[i]] += self._kwargs['alpha'] * (G - self._Q2[states[i]][actions[i]])
        self._updateTargetPolicy(states[i])
        
    def _updateTargetPolicy(self, state: ObsType) -> None:
        self._pi[state].fill(self._epsilon / self._pi[state].shape[0])
        a_ast = np.argmax(self._Q[state] + self._Q2[state])
        self._pi[state][a_ast] += 1 - self._epsilon

if __name__ == '__main__':
    print('only runs in the top-level')