from __future__ import annotations

import gymnasium as gym

import numpy as np

import math
from collections import defaultdict

from src import agent

from typing import TypeVar, SupportsFloat, SupportsInt, Any
from numpy.typing import NDArray

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')

class nStepTemporalDifference(agent.Agent):
    """
    n-Step Temporal Difference agents learn in a n-step bootstrapping sense.
    """
    def __init__(self, env: gym.Env[ObsType, ActType], nStep:SupportsInt, **kwargs: Any) -> None:
        assert(isinstance(env.action_space, gym.spaces.Discrete))

        super().__init__(env, **kwargs)
        if 'epsilonHalfLife' not in self._kwargs:
            self._kwargs['epsilonHalfLife'] = 10000
        if 'alpha' not in self._kwargs:
            self._kwargs['alpha'] = 0.01

        self._Q: defaultdict[ObsType, NDArray] = defaultdict(self._defaultActionValues)
        self._pi: defaultdict[ObsType, NDArray] = defaultdict(self._initTargetPolicy)
        self._nEpisode: int = 0
        self._epsilon: SupportsFloat = 1.0
        self._nStep: SupportsInt = nStep
    
    def _defaultActionValues(self) -> NDArray:
        return np.zeros(self._env.action_space.n)
    
    def _initTargetPolicy(self) -> NDArray:
        n = self._env.action_space.n
        a = np.empty(n)
        a.fill(1.0 / n)
        return a
    
    def _updateActionValue(self, states: NDArray, actions: NDArray, rewards: NDArray, t: SupportsInt, T: SupportsInt) -> None:
        np1 = self._nStep + 1
        G = 0.0
        k = (t + 1) - (self._nStep - 1)
        k_end = min(t + 1, T)
        gamma = 1.0
        while k <= k_end:
            G += gamma * rewards[k % np1]
            k += 1
            gamma *= self._kwargs['gamma']
        if t + 1 < T:
            i = (t + 1) % np1
            G += gamma * self._Q[states[i]][actions[i]]
        i = (t + 1 - self._nStep) % np1
        self._Q[states[i]][actions[i]] += self._kwargs['alpha'] * (G - self._Q[states[i]][actions[i]])
        self._updateTargetPolicy(states[i])
        
    def _trainOnEpisode(self) -> None:
        np1 = self._nStep + 1
        t = 0
        T = np.iinfo(np.int64).max
        st = np.empty(np1, dtype=object)
        at = np.empty(np1, dtype=object)
        rt1 = np.empty(np1)
        st[0], info = self._env.reset(seed=self.seed)
        at[0] = self.takeAction(st[0], info)
        while t <= T + self._nStep - 2:
            if t < T:
                i = (t + 1) % np1
                st[i], rt1[i], terminated, truncated, info = self._env.step(at[t % np1])
                if terminated or truncated:
                    T = t + 1
                else:
                    at[i] = self.takeAction(st[i], info)
            if t >= self._nStep - 1:
                self._updateActionValue(st, at, rt1, t, T)
            t += 1

    def _updateTargetPolicy(self, state: ObsType) -> None:
        self._pi[state].fill(self._epsilon / self._pi[state].shape[0])
        a_ast = np.argmax(self._Q[state])
        self._pi[state][a_ast] += 1 - self._epsilon

    def _epsilonDecay(self) -> None:
        self._epsilon = math.exp(-self._nEpisode*math.log(2)/self._kwargs['epsilonHalfLife'])
    
    def _getParameters(self):
        parameters = super()._getParameters()
        parameters.append('_kwargs')
        parameters.append('_Q')
        parameters.append('_pi')
        parameters.append('_nEpisode')
        parameters.append('_epsilon')
        parameters.append('_nStep')
        return parameters

    def getAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        return np.argmax(self._Q[state])

    def takeAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        return self._rng.choice(self._pi[state].shape[0], p=self._pi[state])
    
    def train(self, nEpisode: SupportsInt) -> None:
        for i in range(nEpisode):
            self._nEpisode += 1
            self._trainOnEpisode()
            self._epsilonDecay()

if __name__ == '__main__':
    print('only runs in the top-level')