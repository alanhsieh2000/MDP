from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces

import numpy as np

import math
import pickle
from collections import defaultdict

from typing import TypeVar, SupportsFloat, SupportsInt, Any
from numpy.typing import NDArray

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')

class Agent:
    """
    The agent interacts with a Gymnasium environment by taking actions and observing rewards. It's goal is to maximize the total rewards generated 
    by the environment after executing actions over time until the terminal state.
    """
    def __init__(self, env: gym.Env[ObsType, ActType], **kwargs: Any) -> None:
        self._env: gym.Env[ObsType, ActType] = env
        self._seed : int | None = None
        self._rng = np.random.default_rng()
        self._kwargs: dict[str, Any] = {'gamma': 1.0}
        for k, v in kwargs.items():
            self._kwargs[k] = v

    def takeAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        """
        It returns the action according to its behavior policy.
        """
        raise NotImplementedError
    
    def getAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        """
        It returns the action according to its target policy.
        """
        raise NotImplementedError
    
    @property
    def seed(self) -> int | None:
        return self._seed
    
    @seed.setter
    def seed(self, sd: int) -> None:
        self._seed = sd
        self._rng = np.random.default_rng(self._seed)

class MonteCarlo(Agent):
    """
    Monte Carlo agents learn in an episode-by-episode sense.
    """
    def __init__(self, env: gym.Env[ObsType, ActType], **kwargs: Any) -> None:
        assert(isinstance(env.action_space, gym.spaces.Discrete))

        super().__init__(env, **kwargs)
        self._kwargs['visit'] = 'every'
        self._kwargs['epsilonHalfLife'] = 10000

        self._sa: list[tuple[ObsType, ActType]] = []
        self._r: list[SupportsFloat] = []
        self._info: list[dict[str, Any]] = []
        self._Q: defaultdict[ObsType, NDArray] = defaultdict(self._defaultActionValues)
        self._n: defaultdict[tuple[ObsType, ActType], SupportsInt] = defaultdict(int)
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
        G = 0
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
        q_ast = max(self._Q[state])
        idx = np.argwhere(self._Q[state] == q_ast)
        self._pi[state].fill(self._epsilon / self._pi[state].shape[0])
        for a_ast in idx:
            self._pi[state][a_ast[0]] += (1 - self._epsilon) / idx.shape[0]

    def _epsilonDecay(self) -> None:
        self._epsilon = math.exp(-self._nEpisode*math.log(2)/self._kwargs['epsilonHalfLife'])

    def getAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        return np.argmax(self._Q[state])

    def takeAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        return self._rng.choice(self._pi[state].shape[0], p=self._pi[state])
    
    def train(self, nEpisode: int) -> None:
        for i in range(nEpisode):
            self._nEpisode += 1
            self._generateAnEpisode()
            self._updateActionValue()
            self._epsilonDecay()

    def save(self, path: str) -> None:
        version = 5
        with open(path, 'wb') as f:
            pickle.dump(self._kwargs, f, protocol=version)
            pickle.dump(self._Q, f, protocol=version)
            pickle.dump(self._n, f, protocol=version)
            pickle.dump(self._pi, f, protocol=version)
            pickle.dump(self._nEpisode, f, protocol=version)
            pickle.dump(self._epsilon, f, protocol=version)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            self._kwargs = pickle.load(f)
            self._Q = pickle.load(f)
            self._n = pickle.load(f)
            self._pi = pickle.load(f)
            self._nEpisode = pickle.load(f)
            self._epsilon = pickle.load(f)

if __name__ == '__main__':
    print('only runs in the top-level')