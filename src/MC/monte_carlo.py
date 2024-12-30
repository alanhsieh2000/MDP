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

class MonteCarlo(agent.Agent):
    """
    Monte Carlo agents learn in an episode-by-episode sense.
    """
    def __init__(self, env: gym.Env[ObsType, ActType], **kwargs: Any) -> None:
        assert(isinstance(env.action_space, gym.spaces.Discrete))

        super().__init__(env, **kwargs)
        if 'visit' not in self._kwargs:
            self._kwargs['visit'] = 'every'
        if 'epsilonHalfLife' not in self._kwargs:
            self._kwargs['epsilonHalfLife'] = 10000

        self._sa: list[tuple[ObsType, ActType]] = []
        self._r: list[SupportsFloat] = []
        self._info: list[dict[str, Any]] = []
        self._Q: defaultdict[ObsType, NDArray] = defaultdict(self._defaultActionValues)
        self._n: defaultdict[tuple[ObsType, ActType], SupportsInt] = defaultdict(int)
        self._pi: defaultdict[ObsType, NDArray] = defaultdict(self._initTargetPolicy)
        self._nEpisode: SupportsInt = 0
        self._epsilon: SupportsFloat = 1.0
    
    def _defaultActionValues(self) -> NDArray:
        return np.zeros(self._env.action_space.n)
    
    def _initTargetPolicy(self) -> NDArray:
        n = self._env.action_space.n
        a = np.empty(n)
        a.fill(1.0 / n)
        return a
    
    def _generateAnEpisode(self) -> None:
        st, info = self._env.reset(seed=self.seed)
        done = False
        while not done:
            at = self.takeAction(st, info)
            self._sa.insert(0, (st, at))
            self._info.insert(0, info)
            st, rt1, terminated, truncated, info = self._env.step(at)
            self._r.insert(0, rt1)
            done = terminated or truncated

    def _updateActionValue(self) -> None:
        G = 0.0
        for sa, r, info, i in zip(self._sa, self._r, self._info, range(len(self._sa))):
            G = r + self._kwargs['gamma'] * G
            if self._kwargs['visit'] == 'first' and sa in self._sa[i+1:]:
                continue
            self._n[sa] += 1
            s, a = sa
            self._Q[s][a] += (G - self._Q[s][a]) / self._n[sa]
            self._updateTargetPolicy(s)
        self._sa = []
        self._r = []
        self._info = []

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
        parameters.append('_n')
        parameters.append('_pi')
        parameters.append('_nEpisode')
        parameters.append('_epsilon')
        return parameters

    def getAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        return np.argmax(self._Q[state])

    def takeAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        return self._rng.choice(self._pi[state].shape[0], p=self._pi[state])
    
    def train(self, nEpisode: SupportsInt) -> None:
        for i in range(nEpisode):
            self._nEpisode += 1
            self._generateAnEpisode()
            self._updateActionValue()
            self._epsilonDecay()

if __name__ == '__main__':
    print('only runs in the top-level')