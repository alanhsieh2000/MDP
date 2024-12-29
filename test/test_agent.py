# test/test_agent.py

import unittest
import gymnasium as gym
from src import agent

class TestAgent(unittest.TestCase):
    def test_init(self):
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = agent.Agent(env, epsilon = 0.1)
        self.assertEqual(agt._env, env)

        self.assertEqual(agt.seed, None)
        agt.seed = 1234
        self.assertEqual(agt.seed, 1234)

        self.assertIn('gamma', agt._kwargs)
        self.assertEqual(agt._kwargs['gamma'], 1.0)
        self.assertIn('epsilon', agt._kwargs)
        self.assertEqual(agt._kwargs['epsilon'], 0.1)

        env.close()

if __name__ == '__main__':
    unittest.main()