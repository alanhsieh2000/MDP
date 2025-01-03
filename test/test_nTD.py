# test/test_nTD.py

import unittest
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from src import nTD

class TestnTD(unittest.TestCase):
    def test_init(self):
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepTemporalDifference(env, nStep)

        self.assertTrue(isinstance(agt._Q[(20, 5, 0)], np.ndarray))
        self.assertEqual(agt._Q[(20, 5, 0)].shape[0], 2)
        self.assertEqual(agt._Q[(20, 5, 0)][0], 0)
        self.assertEqual(agt._Q[(20, 5, 0)][1], 0)

        self.assertTrue(isinstance(agt._pi[(20, 5, 0)], np.ndarray))
        self.assertEqual(agt._pi[(20, 5, 0)].shape[0], 2)
        self.assertEqual(agt._pi[(20, 5, 0)][0], 0.5)
        self.assertEqual(agt._pi[(20, 5, 0)][1], 0.5)

        env.close()

    def test__trainOnEpisode(self):
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepTemporalDifference(env, nStep)
        agt.seed = 1234
        agt._trainOnEpisode()

        self.assertEqual(agt._Q[(17, 10, 0)][0], -0.01)

        env.close()

    def test_train(self):
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepTemporalDifference(env, nStep)
        agt.seed = 1234
        agt.train(1)

        self.assertEqual(agt._nEpisode, 1)
        self.assertEqual(agt._Q[(17, 10, 0)][0], -0.01)
        self.assertEqual(agt._Q[(15, 10, 0)][1], -0.01)
        self.assertEqual(agt._pi[(17, 10, 0)][0], 0.5)
        self.assertEqual(agt._pi[(17, 10, 0)][1], 0.5)
        self.assertEqual(agt._pi[(15, 10, 0)][0], 0.5)
        self.assertEqual(agt._pi[(15, 10, 0)][1], 0.5)
        self.assertEqual(agt._epsilon, 0.9999306876841536)
        
        env.close()

    def test_save_load(self):
        nEpisode = 1000
        nStep = 2
        fname = 'test-nTD.pkl'
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepTemporalDifference(env, nStep)
        agt.train(nEpisode)

        agt.save(fname)

        agt = nTD.nStepTemporalDifference(env, nStep)
        self.assertEqual(agt._nEpisode, 0)
        agt.load(fname)
        self.assertEqual(agt._nEpisode, nEpisode)
        self.assertEqual(agt._nStep, nStep)

        env.close()

    @unittest.skip('save some test time')
    def test_plot(self):
        nEpisode = 100000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=nEpisode)
        agt = nTD.nStepTemporalDifference(env, nStep, epsilonHalfLife=20000)
        agt.train(nEpisode)
        agt.save(f'test-nTD-{nEpisode}.pkl')

        fig, axs = plt.subplots()
        axs.plot(np.convolve(env.return_queue, np.ones(100)/100))
        axs.set_title('Episode Rewards')
        axs.set_xlabel('Episode')
        axs.set_ylabel('Reward')
        plt.savefig('graphics/rewards-nTD.svg', format='svg')

        env.close()

    @unittest.skip('save some test time')
    def test_pi(self):
        nEpisode = 100000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepTemporalDifference(env, nStep)
        agt.load(f'test-nTD-{nEpisode}.pkl')

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
        plt.savefig('graphics/policy-nTD.svg', format='svg')
        env.close()

    @unittest.skip('save some test time')
    def test_Q(self):
        nEpisode = 100000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepTemporalDifference(env, nStep)
        agt.load(f'test-nTD-{nEpisode}.pkl')

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
        plt.savefig('graphics/Q-nTD.svg', format='svg')
        env.close()

    def test_run(self):
        nEpisode = 100000
        nRun = 1000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepTemporalDifference(env, nStep)
        agt.load(f'test-nTD-{nEpisode}.pkl')

        rewards = agt.run(nRun)
        print(f'n-step TD: {nRun} runs, total rewards -> {rewards:.1f}')

        env.close()

    @unittest.skip('save some test time')
    def test_plot_nql(self):
        nEpisode = 100000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=nEpisode)
        agt = nTD.nStepQLearning(env, nStep, epsilonHalfLife=20000)
        agt.train(nEpisode)
        agt.save(f'test-nQL-{nEpisode}.pkl')

        fig, axs = plt.subplots()
        axs.plot(np.convolve(env.return_queue, np.ones(100)/100))
        axs.set_title('Episode Rewards')
        axs.set_xlabel('Episode')
        axs.set_ylabel('Reward')
        plt.savefig('graphics/rewards-nQL.svg', format='svg')

        env.close()

    def test_run_nql(self):
        nEpisode = 100000
        nRun = 1000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepQLearning(env, nStep)
        agt.load(f'test-nQL-{nEpisode}.pkl')

        rewards = agt.run(nRun)
        print(f'n-step Q-Learning: {nRun} runs, total rewards -> {rewards:.1f}')

        env.close()

    @unittest.skip('save some test time')
    def test_plot_nesarsa(self):
        nEpisode = 100000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=nEpisode)
        agt = nTD.nStepExpectedSarsa(env, nStep, epsilonHalfLife=20000)
        agt.train(nEpisode)
        agt.save(f'test-nESarsa-{nEpisode}.pkl')

        fig, axs = plt.subplots()
        axs.plot(np.convolve(env.return_queue, np.ones(100)/100))
        axs.set_title('Episode Rewards')
        axs.set_xlabel('Episode')
        axs.set_ylabel('Reward')
        plt.savefig('graphics/rewards-nESarsa.svg', format='svg')

        env.close()

    def test_run_nesarsa(self):
        nEpisode = 100000
        nRun = 1000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepExpectedSarsa(env, nStep)
        agt.load(f'test-nESarsa-{nEpisode}.pkl')

        rewards = agt.run(nRun)
        print(f'n-step Expected Sarsa: {nRun} runs, total rewards -> {rewards:.1f}')

        env.close()

    @unittest.skip('save some test time')
    def test_plot_ndql(self):
        nEpisode = 100000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=nEpisode)
        agt = nTD.nStepDoubleQLearning(env, nStep, epsilonHalfLife=20000)
        agt.train(nEpisode)
        agt.save(f'test-nDQL-{nEpisode}.pkl')

        fig, axs = plt.subplots()
        axs.plot(np.convolve(env.return_queue, np.ones(100)/100))
        axs.set_title('Episode Rewards')
        axs.set_xlabel('Episode')
        axs.set_ylabel('Reward')
        plt.savefig('graphics/rewards-nDQL.svg', format='svg')

        env.close()

    def test_run_ndql(self):
        nEpisode = 100000
        nRun = 1000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepDoubleQLearning(env, nStep)
        agt.load(f'test-nDQL-{nEpisode}.pkl')

        rewards = agt.run(nRun)
        print(f'n-step double Q-Learning: {nRun} runs, total rewards -> {rewards:.1f}')

        env.close()

    @unittest.skip('save some test time')
    def test_plot_nofftd(self):
        nEpisode = 100000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=nEpisode)
        agt = nTD.nStepOffTemporalDifference(env, nStep)
        agt.train(nEpisode)
        agt.save(f'test-nOffTD-{nEpisode}.pkl')

        fig, axs = plt.subplots()
        axs.plot(np.convolve(env.return_queue, np.ones(100)/100))
        axs.set_title('Episode Rewards')
        axs.set_xlabel('Episode')
        axs.set_ylabel('Reward')
        plt.savefig('graphics/rewards-nOffTD.svg', format='svg')

        env.close()

    def test_run_nofftd(self):
        nEpisode = 100000
        nRun = 1000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepOffTemporalDifference(env, nStep)
        agt.load(f'test-nOffTD-{nEpisode}.pkl')

        rewards = agt.run(nRun)
        print(f'n-step off-policy Sarsa: {nRun} runs, total rewards -> {rewards:.1f}')

        env.close()

    @unittest.skip('save some test time')
    def test_plot_noffql(self):
        nEpisode = 100000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=nEpisode)
        agt = nTD.nStepOffQLearning(env, nStep)
        agt.train(nEpisode)
        agt.save(f'test-nOffQL-{nEpisode}.pkl')

        fig, axs = plt.subplots()
        axs.plot(np.convolve(env.return_queue, np.ones(100)/100))
        axs.set_title('Episode Rewards')
        axs.set_xlabel('Episode')
        axs.set_ylabel('Reward')
        plt.savefig('graphics/rewards-nOffQL.svg', format='svg')

        env.close()

    def test_run_noffql(self):
        nEpisode = 100000
        nRun = 1000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepOffQLearning(env, nStep)
        agt.load(f'test-nOffQL-{nEpisode}.pkl')

        rewards = agt.run(nRun)
        print(f'n-step off-policy Q-Learning: {nRun} runs, total rewards -> {rewards:.1f}')

        env.close()

    @unittest.skip('save some test time')
    def test_plot_noffesarsa(self):
        nEpisode = 100000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=nEpisode)
        agt = nTD.nStepOffExpectedSarsa(env, nStep)
        agt.train(nEpisode)
        agt.save(f'test-nOffESarsa-{nEpisode}.pkl')

        fig, axs = plt.subplots()
        axs.plot(np.convolve(env.return_queue, np.ones(100)/100))
        axs.set_title('Episode Rewards')
        axs.set_xlabel('Episode')
        axs.set_ylabel('Reward')
        plt.savefig('graphics/rewards-nOffESarsa.svg', format='svg')

        env.close()

    def test_run_noffesarsa(self):
        nEpisode = 100000
        nRun = 1000
        nStep = 2
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agt = nTD.nStepOffExpectedSarsa(env, nStep)
        agt.load(f'test-nOffESarsa-{nEpisode}.pkl')

        rewards = agt.run(nRun)
        print(f'n-step off-policy Expected Sarsa: {nRun} runs, total rewards -> {rewards:.1f}')

        env.close()

    @unittest.skip('save some test time')
    def test_off_pi(self):
        nEpisode = 100000
        nStep = 2
        words = ['Without', 'With', 'Target', 'Behavior']
        names = ['nOffTD', 'nOffQL', 'nOffESarsa']
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        agents = [nTD.nStepOffTemporalDifference(env, nStep), 
                  nTD.nStepOffQLearning(env, nStep), 
                  nTD.nStepOffExpectedSarsa(env, nStep)]
        for index in range(len(names)):
            agt = agents[index]
            agt.load(f'test-{names[index]}-{nEpisode}.pkl')

            c = np.zeros((4, 10, 18))
            for k in range(2):
                for i in range(1, 11):
                    for j in range(4, 22):
                        c[k, i-1, j-4] = agt._pi[(j, i, k)][1]
                        c[k+2, i-1, j-4] = agt._b[(j, i, k)][1]

            fig, axs = plt.subplots(4, 1, sharey='all')
            for k in range(4):
                axs[k].set_title(f'{names[index]} {words[(k // 2) + 2]} Policy: {words[k % 2]} usable ace')
                axs[k].set_xlabel('Player sum')
                axs[k].set_ylabel('Dealer showing')
                axs[k].set_xticks(range(18), labels=[x for x in range(4, 22)])
                axs[k].set_yticks(range(10), labels=[y for y in range(1, 11)])
                pos = axs[k].imshow(c[k], cmap='viridis', vmin=0.4, vmax=0.6)
                cbar = fig.colorbar(pos, ax=axs[k], ticks=[0.4, 0.5, 0.6])
                cbar.ax.set_yticklabels(['0.4, stick', '0.5', '0.6, hit'])

            fig.set_figheight(12)
            fig.tight_layout()
            plt.savefig(f'graphics/policy-{names[index]}.svg', format='svg')
            plt.close(fig)

        env.close()

if __name__ == '__main__':
    unittest.main()