from __future__ import annotations

import gymnasium as gym

import numpy as np

import pickle

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

    def _getParameters(self) -> list[Any]:
        """
        It returns model parameters for saving and loading.
        """
        return []

    def _runOnEpisode(self) -> SupportsFloat:
        """
        It returns the discounted total rewards of an episode according to its target policy.
        """
        G = 0.0
        gamma = 1.0
        st, info = self._env.reset(seed=self.seed)
        done = False
        while not done:
            at = self.getAction(st, info)
            st, rt1, terminated, truncated, info = self._env.step(at)
            G += gamma * rt1
            gamma *= self._kwargs['gamma']
            done = terminated or truncated
        return G

    def takeAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        """
        It returns the action according to its behavior policy.
        """
        return self._env.action_space.sample()
    
    def getAction(self, state: ObsType, info: dict[str, Any]) -> ActType:
        """
        It returns the action according to its target policy.
        """
        return self._env.action_space.sample()
    
    def run(self, nRun: SupportsInt) -> SupportsFloat:
        """
        It returns the sum of discounted rewards.
        """
        total = 0.0
        if nRun < 1:
            return total
        
        for i in range(nRun):
            total += self._runOnEpisode()
        return total

    def save(self, path: str) -> None:
        version = 5
        parameters = self._getParameters()
        with open(path, 'wb') as f:
            for k in parameters:
                pickle.dump(self.__dict__[k], f, protocol=version)
    
    def load(self, path: str) -> None:
        parameters = self._getParameters()
        with open(path, 'rb') as f:
            for k in parameters:
                self.__dict__[k] = pickle.load(f)

    @property
    def seed(self) -> int | None:
        return self._seed
    
    @seed.setter
    def seed(self, sd: int) -> None:
        self._seed = sd
        self._rng = np.random.default_rng(self._seed)

if __name__ == '__main__':
    print('only runs in the top-level')