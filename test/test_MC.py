# test/test_MC.py

import unittest
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from src import MC

class TestMonteCarlo(unittest.TestCase):
    def test_init(self):
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = MC.MonteCarlo(env)

        self.assertTrue(isinstance(agt._Q[(20, 5, 0)], np.ndarray))
        self.assertEqual(agt._Q[(20, 5, 0)].shape[0], 2)
        self.assertEqual(agt._Q[(20, 5, 0)][0], 0)
        self.assertEqual(agt._Q[(20, 5, 0)][1], 0)

        self.assertTrue(isinstance(agt._pi[(20, 5, 0)], np.ndarray))
        self.assertEqual(agt._pi[(20, 5, 0)].shape[0], 2)
        self.assertEqual(agt._pi[(20, 5, 0)][0], 0.5)
        self.assertEqual(agt._pi[(20, 5, 0)][1], 0.5)

        env.close()

    def test__generateAnEpisode(self):
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = MC.MonteCarlo(env)
        agt.seed = 1234
        agt._generateAnEpisode()

        self.assertEqual(agt._r, [-1.0, 0.0])
        self.assertEqual(agt._sa, [((17, 10, 0), 0), ((15, 10, 0), 1)])
        
        env.close()

    def test_train(self):
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = MC.MonteCarlo(env)
        agt.seed = 1234
        agt.train(1)

        self.assertEqual(agt._nEpisode, 1)
        self.assertEqual(agt._n[((17, 10, 0), 0)], 1)
        self.assertEqual(agt._n[((15, 10, 0), 1)], 1)
        self.assertEqual(agt._Q[(17, 10, 0)][0], -1.0)
        self.assertEqual(agt._Q[(15, 10, 0)][1], -1.0)
        self.assertEqual(agt._pi[(17, 10, 0)][0], 0.5)
        self.assertEqual(agt._pi[(17, 10, 0)][1], 0.5)
        self.assertEqual(agt._pi[(15, 10, 0)][0], 0.5)
        self.assertEqual(agt._pi[(15, 10, 0)][1], 0.5)
        self.assertEqual(agt._epsilon, 0.9999306876841536)
        
        env.close()

    def test_save_load(self):
        nEpisode = 1000
        fname = 'test-MC.pkl'
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = MC.MonteCarlo(env)
        agt.train(nEpisode)

        agt.save(fname)

        agt = MC.MonteCarlo(env)
        self.assertEqual(agt._nEpisode, 0)
        agt.load(fname)
        self.assertEqual(agt._nEpisode, nEpisode)

        env.close()

    @unittest.skip('save some test time')
    def test_plot(self):
        nEpisode = 100000
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=nEpisode)
        agt = MC.MonteCarlo(env, epsilonHalfLife=20000)
        agt.train(nEpisode)
        agt.save(f'test-MC-{nEpisode}.pkl')

        fig, axs = plt.subplots()
        axs.plot(np.convolve(env.return_queue, np.ones(100)/100))
        axs.set_title('Episode Rewards')
        axs.set_xlabel('Episode')
        axs.set_ylabel('Reward')
        plt.savefig('rewards-MC.svg', format='svg')

        env.close()

    @unittest.skip('save some test time')
    def test_pi(self):
        nEpisode = 100000
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = MC.MonteCarlo(env)
        agt.load(f'test-MC-{nEpisode}.pkl')

        c = np.zeros((2, 10, 18))
        for k in range(2):
            for i in range(1, 11):
                for j in range(4, 22):
                    c[k, i-1, j-4] = agt.getAction((j, i, k), {})
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey='all')

        ax1.set_title('Policy: Without usable ace')
        ax1.set_xlabel('Player sum')
        ax1.set_ylabel('Dealer showing')
        ax1.set_xticks(range(18), labels=[x for x in range(4, 22)])
        ax1.set_yticks(range(10), labels=[y for y in range(1, 11)])
        ax1.imshow(c[0])

        ax2.set_title('Policy: With usable ace')
        ax2.set_xlabel('Player sum')
        ax2.set_ylabel('Dealer showing')
        ax2.set_xticks(range(18), labels=[x for x in range(4, 22)])
        ax2.imshow(c[1])

        hit = mpatches.Patch(color='yellow', label='hit')
        stick = mpatches.Patch(color='purple', label='stick')
        ax2.legend(handles=[hit, stick], loc='upper left')

        fig.set_figheight(6)
        fig.tight_layout()
        plt.savefig('policy-MC.svg', format='svg')
        env.close()

    @unittest.skip('save some test time')
    def test_n(self):
        nEpisode = 100000
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = MC.MonteCarlo(env)
        agt.load(f'test-MC-{nEpisode}.pkl')

        c = np.zeros((2, 2, 10, 18))
        words = ['Without', 'With', 'stick', 'hit']
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

        for a in range(2):
            for k in range(2):
                for i in range(1, 11):
                    for j in range(4, 22):
                        c[a, k, i-1, j-4] = agt._n[((j, i, k), a)]
                tmp1 = words[k]
                tmp2 = words[a+2]
                im = axs[a, k].imshow(c[a, k, :, :])
                axs[a, k].set_title(f'n: {tmp1} usable ace for {tmp2}')
                axs[a, k].set_xlabel('Player sum')
                axs[a, k].set_xticks(range(18), labels=[x for x in range(4, 22)])
                axs[a, k].set_ylabel('Dealer showing')
                axs[a, k].set_yticks(range(10), labels=[y for y in range(1, 11)])

        fig.set_figheight(6)
        fig.set_figwidth(10)
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.75])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig('n.svg', format='svg')
        env.close()

    @unittest.skip('save some test time')
    def test_Q(self):
        nEpisode = 100000
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = MC.MonteCarlo(env)
        agt.load(f'test-MC-{nEpisode}.pkl')

        c = np.zeros((2, 2, 10, 18))
        words = ['Without', 'With', 'stick', 'hit']
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

        for a in range(2):
            for k in range(2):
                for i in range(1, 11):
                    for j in range(4, 22):
                        c[a, k, i-1, j-4] = agt._Q[(j, i, k)][a]
                tmp1 = words[k]
                tmp2 = words[a+2]
                im = axs[a, k].imshow(c[a, k, :, :])
                axs[a, k].set_title(f'Q: {tmp1} usable ace for {tmp2}')
                axs[a, k].set_xlabel('Player sum')
                axs[a, k].set_xticks(range(18), labels=[x for x in range(4, 22)])
                axs[a, k].set_ylabel('Dealer showing')
                axs[a, k].set_yticks(range(10), labels=[y for y in range(1, 11)])

        fig.set_figheight(6)
        fig.set_figwidth(10)
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.75])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig('Q-MC.svg', format='svg')
        env.close()

    def test_run(self):
        nEpisode = 100000
        nRun = 1000
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = MC.MonteCarlo(env)
        agt.load(f'test-MC-{nEpisode}.pkl')

        rewards = agt.run(nRun)
        print(f'MC: {nRun} runs, total rewards -> {rewards:.1f}')

        env.close()

if __name__ == '__main__':
    unittest.main()