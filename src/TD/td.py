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

class TemporalDifference(agent.Agent):
    """
    Temporal Difference agents learn in a bootstrapping sense.
    """
    def __init__(self, env: gym.Env[ObsType, ActType], **kwargs: Any) -> None:
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
    
    def _defaultActionValues(self) -> NDArray:
        return np.zeros(self._env.action_space.n)
    
    def _initTargetPolicy(self) -> NDArray:
        n = self._env.action_space.n
        a = np.empty(n)
        a.fill(1.0 / n)
        return a
    
    def _updateActionValue(self, state: ObsType, action: ActType, reward: SupportsFloat, nextState: ObsType, nextAction: ActType) -> None:
        self._Q[state][action] += self._kwargs['alpha'] * (reward + self._kwargs['gamma'] * self._Q[nextState][nextAction] - self._Q[state][action])
        self._updateTargetPolicy(state)
        
    def _trainOnEpisode(self) -> None:
        st, info = self._env.reset(seed=self.seed)
        at = self.takeAction(st, info)
        done = False
        while not done:
            st1, rt1, terminated, truncated, info = self._env.step(at)
            at1 = self.takeAction(st1, info)
            self._updateActionValue(st, at, rt1, st1, at1)
            st = st1
            at = at1
            done = terminated or truncated

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
        return parameters

    def getAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        return np.argmax(self._Q[state])

    def takeAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        return self._rng.choice(self._pi[state].shape[0], p=self._pi[state])
    
    def train(self, nEpisode: int) -> None:
        for i in range(nEpisode):
            self._nEpisode += 1
            self._trainOnEpisode()
            self._epsilonDecay()

if __name__ == '__main__':
    print('only runs in the top-level')