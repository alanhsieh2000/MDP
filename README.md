# Setup
The easiest way to try this code repository is to run it in a pre-built docker container and to connect the development container through Visual 
Studio Code and Dev Container extension. The Dockerfile and devcontainer.json needed for this setup locates in the setup folder, and docker 
commands to build the docker image and to create the named volume are as the below.

## Docker Container
```
docker build -t alan/mdp:1.0.0 .
```

## Docker Volume
```
docker volume create vol-workspace
```

## Unit Test
After opening a folder or workspace in the development container, clone this git repository and change to the MDP folder. If everything works 
well, we can run the following unit test successfully.
```
/workspace/MDP # python -m unittest -v
```

# MonteCarlo
The MonteCarlo class implements an on-policy &epsilon;-Greedy tabular method. Both state and action space has to be discrete. Gymnasium's Toy 
Text environments like Blackjack, Taxi, Cliff Walking and Frozen Lake could be solved by this class.

## How to use
Its &epsilon; decays exponentially, and the decay rate can be controlled by keyword arguments of MonteCarlo constructor *epsilonHalfLife*. 
Both first-visit and every-visit variants of Monte Carlo method are supported. It can be controlled by keyword argument *visit* being 'first' 
or 'every' (default).
```
import gymnasium as gym
from src import MC

# training
env = gym.make('Blackjack-v1', natural=False, sab=False)
agt = MC.MonteCarlo(env)
agt.train(10000)

# save the trained model
agt.save('mc.pkl')

# testing - greedy
state, info = env.reset()
action = agt.getAction(state, info)

# load the trained model
agt.load('mc.pkl')
```

## Example - Blackjack
As shown in Figure 1, the moving average of total episode rewards get improved. The more episodes it learns from, the higher average reward 
the target policy can lead to. 

![Figure 1](graphics/rewards-MC.svg)

The learned target policy is shown in Figure 2. In some states, obviously, the best action shouldn't be 'stick'. 

![Figure 2](graphics/policy-MC.svg)

Looking into the private member variable _n and _Q as shown in Figure 3 and 4, we can see those states are seldom or never visited. It's a 
clue of insufficient training episodes.

![Figure 3](graphics/n.svg)

![Figure 4](graphics/Q-MC.svg)

Figure 1 ~ 4 can be helpful tools to understand whether more training is needed. The code to plot them is included in test cases.

# Temporal Difference
The TemporalDifference class implements an on-policy TD(0) - Sarsa with &epsilon;-Greedy tabular method. Both state and action space has to 
be discrete. Gymnasium's Toy Text environments like Blackjack, Taxi, Cliff Walking and Frozen Lake could be solved by this class.

## How to use
Its &epsilon; decays exponentially. Its decay rate can be controlled by keyword arguments of TemporalDifference constructor *epsilonHalfLife*, 
and its learning rate can be controlled by keyword argument *alpha*. 
```
import gymnasium as gym
from src import TD

# training
env = gym.make('Blackjack-v1', natural=False, sab=False)
agt = TD.TemporalDifference(env)
agt.train(10000)

# save the trained model
agt.save('td.pkl')

# testing - greedy
state, info = env.reset()
action = agt.getAction(state, info)

# load the trained model
agt.load('td.pkl')
```

## Example - Blackjack
As shown in Figure 5, the moving average of total episode rewards get improved. The more episodes it learns from, the higher average reward 
the target policy can lead to. 

![Figure 5](graphics/rewards-TD.svg)

The learned target policy by the Temporal Difference mathod, one step TD or TD(0), is shown in Figure 6. 

![Figure 6](graphics/policy-TD.svg)

![Figure 7](graphics/Q-TD.svg)

# n-step Temporal Difference
The nStepTemporalDifference class implements an on-policy TD - n-step Sarsa with &epsilon;-Greedy tabular method. Both state and action space 
has to be discrete. Gymnasium's Toy Text environments like Blackjack, Taxi, Cliff Walking and Frozen Lake could be solved by this class. Unlike 
the Monte Carlo method, n-step TD doesn't have to wait for the termination of episodes. It bootstraps. And unlike the TD(0) method, n-step TD 
doesn't have to estimate the state-action value only relying on 1-step rewards. The Monte Carlo and TD(0) methods locate at the two extreme ends, 
while the n-step TD method is in the middle. n-TD is the same as TD(0) if n is 1, and n-TD is the same as Monte Carlo if n is infinite.

## How to use
The step size can be assigned by the 2nd positional argument nStep. Its &epsilon; decays exponentially. Its decay rate can be controlled by 
keyword arguments of nStepTemporalDifference constructor *epsilonHalfLife*, and its learning rate can be controlled by keyword argument *alpha*. 
```
import gymnasium as gym
from src import nTD

# training
nStep = 2
env = gym.make('Blackjack-v1', natural=False, sab=False)
agt = nTD.nStepTemporalDifference(env, nStep)
agt.train(10000)

# save the trained model
agt.save('ntd.pkl')

# testing - greedy
state, info = env.reset()
action = agt.getAction(state, info)

# load the trained model
agt.load('ntd.pkl')
```

## Example - Blackjack

![Figure 8](graphics/rewards-nTD.svg)

The learned target policy by the final Temporal Difference method, n-step TD, is shown in Figure 9. 

![Figure 9](graphics/policy-nTD.svg)

![Figure 10](graphics/Q-nTD.svg)

# Other n-step Temporal Difference methods
Some other n-step on-policy Temporal Difference methods, including Q-Learning, expected Sarsa and double Q-Learning, and off-policy Temporal 
Difference methods, including Sarsa, Q-Learning and expected Sarsa, are also in the src/nTD folder. For off-policy methods, the behavior policy 
is &epsilon;-soft, and the target policy is &epsilon;-greedy.

## n-step off-policy Temporal Difference - Sarsa
Figure 11 shows the differences between the target and behavior policies.

![Figure 11](graphics/policy-nOffTD.svg)

# Reference
- Carnegie Mellon University, Fragkiadaki, Katerina, et al. 2024. "10-403 Deep Reinforcement Learning" As of 8 November, 2024. 
https://cmudeeprl.github.io/403website_s24/.
- Sutton, Richard S., and Barto, Andrew G. 2018. Reinforcement Learning - An indroduction, second edition. The MIT Press.
- Towers, et al. 2024. "Gymnasium: A Standard Interface for Reinforcement Learning Environments", [arXiv:2407.17032](https://arxiv.org/abs/2407.17032).
- Farama Foundation, n.d. Gymnasium. https://github.com/farama-Foundation/gymnasium?tab=readme-ov-file
