# MDP
Markov decision processes, assumes all the information of the history is encoded in the latest state. Therefore, the decision to make at time 
t can only depend on the state at time t. Unlike the bandits problem, which is stateless, states of a MDP problem matter.

There are some definitions and notations used in this document. In general, random variables are in uppercase, and samples of these random 
variables are in lowercase (Sutton et al., 2018).

State: $S_t$ and $s_t$, $\forall s \in S$

Action: $A_t$ and $a_t$, $\forall a \in A$

Reward: $R_t$ and $r_t$, $\forall r \in R$

Episode: A sequence or trajectory $S_0$, $A_0$, $R_1$, $S_1$, $A_1$, $R_2$, ..., $S_T$, where $S_T$ is the terminal state.

Environment's dynamics: A conditional PDF $p(s_{t+1}, r_{t+1} \vert s_t, a_t)$

$${\sum_{s_{t+1} \in S} \sum_{r_{t+1} \in R} p(s_{t+1}, r_{t+1} \vert s_t, a_t)=1, \forall s_t \in S, a_t \in A}$$

Discount rate: The reward in the next time period values less than the current reward by $\gamma$, $0 \leq \gamma \leq 1$

Return: $G_t$

$${\begin{align*}
  G_t &\doteq \sum_{k=0}^{T-t-1} \gamma^k R_{k+t+1} \\ 
  &= R_{t+1} + \gamma \cdot G_{t+1} &(1) \\
  G_T &\doteq 0 \\
  &t \in [0, T-1]
\end{align*}}$$

Policy: A mapping from states to probability of selecting each possible action $\pi(a_t \vert s_t)$

Value function: The expected return when starting from $s_t$ and following $\pi$ thereafter, $v_\pi(s_t)$

$${\begin{align*}
  v_\pi(s_t) &\doteq E_\pi[G_t \vert S_t = s_t] \\
  &= E_\pi[R_{t+1} + \gamma \cdot G_{t+1} \vert S_t = s_t] &(2) \\
  &= \sum_{a_t} \pi(a_t \vert s_t) \cdot \sum_{s_{t+1}, r_{t+1}} p(s_{t+1}, r_{t+1} \vert s_t, a_t) \cdot [r_{t+1} + \gamma \cdot v_\pi(s_{t+1})] &(3) \\
  &t \in [0, T-1] 
\end{align*}}$$

Action-value function: The expected return when starting from $s_t$, and taking $a_t$, and also following $\pi$ thereafter, $q_\pi(s_t)$

$${\begin{align*}
  q_\pi(s_t, a_t) &\doteq E_\pi[G_t \vert S_t = s_t, A_t = a_t] \\
  &= E_\pi[R_{t+1} + \gamma \cdot G_{t+1} \vert S_t = s_t, A_t = a_t] &(4) \\
  &= \sum_{s_{t+1}, r_{t+1}} p(s_{t+1}, r_{t+1} \vert s_t, a_t) \cdot [r_{t+1} + \gamma \cdot v_\pi(s_{t+1})] \\
  &= \sum_{s_{t+1}, r_{t+1}} p(s_{t+1}, r_{t+1} \vert s_t, a_t) \cdot [r_{t+1} + \gamma \cdot \sum_{a_{t+1}} \pi(a_{t+1} \vert s_{t+1}) \cdot q_\pi(s_{t+1}, a_{t+1})] &(5) \\
  &t \in [0, T-1]
\end{align*}}$$

Optimal policy: A policy better than or equal to all the other policies, $\pi_\ast$. By following it, its coresponding value function and 
action-value function are both optimal.

Optimal value function: The value function following $\pi_\ast$

$${v_\ast(s_t) \doteq \max_{\pi} v_\pi(s_t), \forall s_t \in S}$$

Optimal action-value function: The action-value function following $\pi_\ast$

$${q_\ast(s_t, a_t) \doteq \max_{\pi} q_\pi(s_t, a_t), \forall s_t \in S, a_t \in A}$$

Estimated value function: $V(s_t)$, the estimation of $v_\pi(s_t)$.

Estimated action-value function: $Q(s_t, a_t)$, the estimation of $q_\pi(s_t, a_t)$.

$v_\pi(s_t)$ tells us what the value of each state is, the prediction, and $q_\pi(s_t, a_t)$ tells us how to choose the right action of each 
state, the control. The goal of solving a MDP problem is to find the $\pi_\ast$, or to approximate it.

# Dynamic Programming
If the environment's dynamics are known, there is nothing more to explore or to learn. Under the policy $\pi(a_t \vert s_t)$, equation (3) 
give us a linear equation for each state, $T$ in total. We can then use $T$ linear equations to solve $T$ unknowns, $v_\pi(s_0)$, ..., 
$v_\pi(s_{T-1})$. However, when $T$ is a large number, using DP is a better option.

Consider a sequence of approximate value functions $v_0, v_1, ..., v_k$, where $v_0$ is chosen arbitrarily. By using the following update 
rule, 

$${v_{i+1}(s_t) = \sum_{a_t} \pi(a_t \vert s_t) \cdot \sum_{s_{t+1}, r_{t+1}} p(s_{t+1}, r_{t+1} \vert s_t, a_t) \cdot [r_{t+1} + \gamma \cdot v_i(s_{t+1})]}$$

then

$${\lim_{k \to \infty} v_k(s_t)=v_\pi(s_t)}$$

This algorithm is called *iterative policy evaluation*. We calculate the sequence starting from $v_1(s_t)$ for each state, up to $v_k(s_t) 
also for each state, until $v_k(s_t) - v_{k-1}(s_t)$ is less than a small number $\theta$ for each state.

Before we move forward, we need to introduce a confusing notation $\pi^\prime(s_t)$. Unlike $\pi(a_t \vert s_t)$, which is a conditional PDF, 
$\pi^\prime(s_t)$ returns the best action in state $s_t$.

$${\begin{align*}
  \pi^\prime(s_t) &\doteq \arg \max_{a_t} q_\pi(s_t, a_t) \\
  &= \arg \max_{a_t} \sum_{s_{t+1}, r_{t+1}} p(s_{t+1}, r_{t+1} \vert s_t, a_t) \cdot [r_{t+1} + \gamma \cdot v_\pi(s_{t+1})] &(6)
\end{align*}}$$

Therefore, by the definition of $v_\ast(s_t)$ and equation (3), we have

$${\begin{align*}
  v_\ast(s_t) &= \max_\pi v_\pi(s_t) \\
  &= \sum_{a_t} \pi_\ast(a_t \vert s_t) \cdot \sum_{s_{t+1}, r_{t+1}} p(s_{t+1}, r_{t+1} \vert s_t, a_t) \cdot [r_{t+1} + \gamma \cdot v_\ast(s_{t+1})] \\
  &= \sum_{s_{t+1}, r_{t+1}} p(s_{t+1}, r_{t+1} \vert s_t, \pi_\ast(s_t)) \cdot [r_{t+1} + \gamma \cdot v_\ast(s_{t+1})]
\end{align*}}$$

The update rule we will use is 

$${\begin{align*}
  v_{i+1}(s_t) &= \sum_{s_{t+1}, r_{t+1}} p(s_{t+1}, r_{t+1} \vert s_t, \pi^\prime(s_t)) \cdot [r_{t+1} + \gamma \cdot v_i(s_{t+1})] &(7) \\
  & \to v_\ast(s_t), \text{when } k \to \infty, \pi^\prime(s_t) \to \pi_\ast(s_t)
\end{align*}}$$

We start from an arbitary $\pi_0(s_t)$ and $v_0(s_t)$. Alternately and iteratively, we use equation (7) to calculate $v_1(s_t)$, which is called 
policy **E**valuation, and then use equation (6) to update $\pi_1(s_t)$, which is called policy **I**mprovement. As the iteration continues, 
$\pi_k(s_t)$ and $v_k(s_t)$ approximate $\pi_\ast(s_t)$ and $v_\ast(s_t)$.

$${\pi_0 \to^E v_1 \to^I \pi_1 \to^E v_2 \to^I \pi_2 \to^E ... \to^I \pi_\ast \to^E v_\ast}$$

Here is the final algorithm.

1. Initialization 
    - initialize $\pi(s)$ and $v(s)$ arbitarily for all $s$.
    - $v(s_T) = 0$
    - set $\theta$ to an acceptable error of $\vert v(s) - v_\ast(s) \vert$.
2. Policy Evaluation
    - repeat
        - $\Delta = 0$
        - for each $s$ excluding $s_T$:
            - $v_{old} = v(s)$
            - $v(s) = \sum_{s^\prime, r} p(s^\prime, r \vert s, \pi(s)) \cdot [r + \gamma \cdot v(s^\prime)]$
            - $\Delta = \max (\Delta, \vert v_{old} - v(s) \vert)$
    - until $\Delta \le \theta$
3. Policy Improvement
    - policy_stable = True
    - for each $s$ excluding $s_T$:
        - $a_{old} = \pi(s)$
        - $\pi(s) = \arg \max_a \sum_{s^\prime, r} p(s^\prime, r \vert s, \pi(s)) \cdot [r + \gamma \cdot v(s^\prime)]$
        - if $a_{old} \neq \pi(s)$
            - policy_stable = False
    - if policy_stable
        - return $v(s), \pi(s)$
    - else
        - go to step 2

# Monte Carlo Learning
Most environment's dynamics of MDP problems are unknown, at least to be known at a high cost. In this case, we need to learn them by 
exploration. MC methods learn the action-value function and the policy from experiences of sampled sequences or trajectories, episodic tasks 
that have terminal states, and use the simplest possible idea to estimate action-value, the mean return.

$${\text{An episode: }s_0, a_0, r_1, s_1, a_1, r_2, ..., s_{T-1}, a_{T-1}, r_T, s_T}$$

Although we can use MC methods to estimate value function as well by using equation (2), we can not use the learned value function and plug 
it into equation (6) to improve a policy like DP does without the knowledge of environment's dynamics. That's why we need to learn 
$q_\ast(s_t, a_t)$ instead. In other words, we start from an arbitary $\pi_0(s_t)$ and $q_0(s_t, a_t)$. Alternately and iteratively, we 
calculate $q_1(s_t, a_t)$, which is also called policy **E**valuation, and then update $\pi_1(s_t)$, which is called policy **I**mprovement. 
As the iteration continues, $\pi_k(s_t)$ and $q_k(s_t, a_t)$ approximate $\pi_\ast(s_t)$ and $q_\ast(s_t, a_t)$.

$${\pi_0 \to^E q_1 \to^I \pi_1 \to^E q_2 \to^I \pi_2 \to^E ... \to^I \pi_\ast \to^E q_\ast}$$

There are some key assumptions:

1. Episodes have exploring starts, which means $s_0$ and $a_0$ are randomly chosen.
    - If we choose a deterministic way to improve the policy, like greedy as in DP, state-action pairs are not randomly chosen and will unlikey 
    cover all possible state-action pairs in the long run. 
    - Convergence of ES has not yet been proven (Sutton et al., 2018, p.99).
    - To remove this assumption, we have on-policy and off-policy methods, evaluating and improving the policy used in generating episodes and 
    evaluating and improving the policy different from that used in generating episodes.
2. Policy evaluation could be done with an infinite number of episodes.
    - It would be too long for each $q_i$ to converge under $\pi_{i-1}$.
    - We don't complete policy evaluation before returing to policy improvement. Extremely, we move $q_i$ toward $q_\pi$ by 1 episode and then 
    update $\pi_i$. Even $\pi_i$ may still be the same as $\pi_{i-1}$, unlike DP, we continue iterating.

## On-policy: &epsilon;-greedy
A policy is *soft*, meaning that $\pi(a_t \vert s_t) \gt 0, \forall s_t \in S, a_t \in A$. By this definition, &epsilon;-greedy algorithm 
is soft, which explores non-greedy actions at a probability of &epsilon;, and the rest of probability 1-&epsilon; goes to the greedy action. 
&epsilon;-greedy algorithm explores every non-greedy action at a probability of $\frac{\epsilon}{\vert A(S_t) \vert}$ equally at least. Since 
the sum of probabilities of all possible actions equals to 1, it takes the greedy action at a probability of $1 - \epsilon - 
\frac{\epsilon}{\vert A(S_t) \vert}$ if there is only one greedy action.

1. Initialize
    - $\pi$ has arbitrary $A^\ast$ for all states, with &epsilon;-greedy
    - fill Q(s, a) with arbitrary real number
    - fill n(s, a) with 0
2. Iterate
    - Generate an episode following $\pi$: $s_0, a_0, r_1, ..., s_{T-1}, a_{T-1}, r_T, s_T$
    - G = 0
    - for t = T-1, T-2, ..., 0:
        - G = $r_{t+1} + \gamma \cdot G$
        - if every-visit or (first-visit and $s_t, a_t$ is the first visit in the episode):
            - n($s_t$, $a_t$) += 1
            - Q($s_t$, $a_t$) += (1 / n($s_t$, $a_t$)) * (G - Q($s_t$, $a_t$))
            - $A^\ast$ = $\arg \max_a Q(s_t, a)$
            - $\pi$.setGreedyAction($s_t$, $A^\ast$)

$${\pi(a_t \vert s_t)=\begin{cases}
  1 - \epsilon + \frac{\epsilon}{\vert A \vert}, &\text{if }a_t = A^\ast \\
  \frac{\epsilon}{\vert A \vert}, &\text{if }a_t \neq A^\ast
\end{cases}}$$

## Off-policy: importance sampling
There are 2 policies used in off-policy methods, the target policy $\pi$ which is used for policy evaluation and improvement, and the behavior 
policy $b$ which is used for episode generation. Off-policy is more general than on-policy because it includes on-policy. To be specific, 
on-policy is a special case of off-policy when $b = \pi$. 

Off-policy methods have an assumption, the assumption of coverage, which requires $b(a_t \vert s_t) > 0$ if $\pi(a_t \vert s_t) > 0, \forall s_t \in S$. 
A simple choice of $\pi$ being greedy and $b$ being &epsilon;-soft would meet this requirement.

Consider an episode starting at $S_t$, $A_t, S_{t+1}, A_{t+1}, ..., S_T$, occurring under policy $\pi$. The probability of the episode is

$${\begin{align*}
  &\Pr(A_t, S_{t+1}, A_{t+1}, ..., S_T \vert S_t) \text{ where } A_{t:T-1} \thicksim \pi \\
  &= \pi(A_t \vert S_t)p(S_{t+1} \vert S_t, A_t)\pi(A_{t+1} \vert S_{t+1})...p(S_T \vert S_{T-1}, A_{T-1}) \\
  &= \prod_{k=t}^{T-1} \pi(A_k \vert S_k)p(S_{k+1} \vert S_k, A_k)
\end{align*}}$$

For the same reason, the probability of the same episode, occurring under policy $b$ is $\prod_{k=t}^{T-1} b(A_k \vert S_k)p(S_{k+1} \vert S_k, A_k)$. 
The importance-sampling ratio is defined as

$${\rho_{t:T-1} \doteq \frac{\prod_{k=t}^{T-1} \pi(A_k \vert S_k)p(S_{k+1} \vert S_k, A_k)}{\prod_{k=t}^{T-1} b(A_k \vert S_k)p(S_{k+1} \vert S_k, A_k)} 
= \prod_{k=t}^{T-1} \frac{\pi(A_k \vert S_k)}{b(A_k \vert S_k)}}$$

To get $v_\pi(s_t)$ or $q_\pi(s_t, a_t)$, we can simply scale returns occurring under the policy $b$ and average the results, with respect to 
states or state-action pairs. There are 2 versions of importance sampling: ordinary importance sampling and weighted importance sampling. 

$${\begin{align*}
  Q_{ordinary}(s, a) &\doteq \frac{\sum_{i=1}^{N}\rho_{t_i:T_i-1} \cdot G_{t_i}}{N}, \\
  Q_{weighted}(s, a) &\doteq \frac{\sum_{i=1}^{N}\rho_{t_i:T_i-1} \cdot G_{t_i}}{\sum_{i=1}^{N}\rho_{t_i:T_i-1}}, \\
  &\text{where $N$ is the number of episodes}
\end{align*}}$$

Let $W_i = \rho_{t_i:T_i-1}$. The update rule for the ordinary importance sampling is simply

$${\begin{align*}
  Q_{n,ordinary}(s, a) &\doteq \frac{\sum_{k=1}^{n-1}W_k G_k}{n-1}, \\
  Q_{n+1,ordinary}(s, a) &\doteq \frac{\sum_{k=1}^{n}W_k G_k}{n} \\
  &= \frac{\sum_{k=1}^{n-1}W_k G_k + W_n G_n}{n} \\
  &= \frac{(n-1) Q_{n,ordinary}(s, a) + W_n G_n}{n} \\
  &= Q_{n,ordinary}(s, a) + \frac{1}{n} (W_n G_n - Q_{n,ordinary}(s, a)), \\
  &\text{where } n \ge 1
\end{align*}}$$

Let $C_n = \sum_{k=1}^{n} W_k$. The update rule for the weighted importance sampling is

$${\begin{align*}
  C_n &\doteq \sum_{k=1}^{n} = C_{n-1} + W_n \\
  Q_{n,weighted}(s, a) &\doteq \frac{\sum_{k=1}^{n-1}W_k G_k}{\sum_{k=1}^{n-1}W_k} \\
  &= \frac{\sum_{k=1}^{n-1}W_k G_k}{C_n - W_n}, \\
  Q_{n+1,weighted}(s, a) &\doteq \frac{\sum_{k=1}^{n}W_k G_k}{\sum_{k=1}^{n}W_k} \\
  &= \frac{\sum_{k=1}^{n-1}W_k G_k + W_n G_n}{C_n} \\
  &= \frac{(C_n - W_n) Q_{n,weighted}(s, a) + W_n G_n}{C_n} \\
  &= Q_{n,weighted}(s, a) + \frac{W_n}{C_n} (G_n - Q_{n,weighted}(s, a)), \\
  &\text{where } n \ge 1, C_0 = 0
\end{align*}}$$

1. Initialize
    - fill Q($s_t$, $a_t$) with arbitrary real number
    - $\pi(s_t)$ = $\arg \max_a q(s_t, a_t)$, the greedy policy
    - if it is ordinary:
        - fill n($s_t$, $a_t$) with 0
    - if it is weighted:
        - fill C($s_t$, $a_t$) with 0.0
2. Iterate
    - b = any soft policy
    - Generate an episode following $b: s_0, a_0, r_1, ..., s_{T-1}, a_{T-1}, r_T$
    - G = 0
    - W = 1
    - for t = T-1, T-2, ..., 0, while W != 0:
        - G = $r_{t+1} + \gamma \cdot G$
        - if it is ordinary:
            - n($s_t$, $a_t$) += 1
            - Q($s_t$, $a_t$) += (1 / n($s_t$, $a_t$)) * (W * G - Q($s_t$, $a_t$))
        - if it is weighted:
            - C($s_t$, $a_t$) += W
            - Q($s_t$, $a_t$) += (W / C($s_t$, $a_t$)) * (G - Q($s_t$, $a_t$))
        - $\pi(s_t)$ = $\arg \max_a q(s_t, a_t)$
        - if $a_t$ != $\pi(s_t)$:
            - break (proceed to the next episode)
        - W *= 1 / $b(a_t \vert s_t)$ ($\pi(a_t \vert s_t) = 1$ because it is greedy)

# One-step Temporal Difference Learning
DP uses equation (3) to calculate values, and MC uses sampled rewards and returns to approximate equation (4) for values. Because we won't 
know what returns are before the termination of episodes, MC needs to complete an episode to start its learning. The feature of learning without 
waiting for the termination of episodes is called bootstrapping. Bootstrapping is the advantage of DP. On the other hand, the advantage of MC 
is no prior knowledge of environment's dynamics required. By replacing sampled returns with estimated value of the next state or state-action 
pair to approximate equation (4), TD has both advantages.

## on-policy: Sarsa
The name Sarsa comes after the pattern of episodes, $S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}$. Before it updates $Q(s_t, a_t)$, Sarsa takes the 
action under the policy $\pi(a_t \vert s_t)$, observes the sampled reward $r_{t+1}$ and state $s_{t+1}$, and consults the same policy 
$\pi(a_{t+1} \vert s_{t+1})$ for what $a_{t+1}$ should be. The policy can be &epsilon;-greedy or &epsilon;-soft. The update rule is

$${Q_{n+1}(s_t, a_t) = Q_n(s_t, a_t) + \alpha \cdot [r_{t+1} + \gamma \cdot Q_{n}(s_{t+1}, a_{t+1}) - Q_n(s_t, a_t)], \alpha \in (0, 1]}$$

1. Initialize
    - fill Q($s_t$, $a_t$) with arbitrary real number
    - Q($s_T$, -) = 0
2. Iterate
    - generate the initial state $s_0$ of an episode
    - $s_t$ = $s_0$
    - $a_t$ = $\pi(s_t)$
    - repeat
        - take action $a_t$
        - observe $r_{t+1}$, $s_{t+1}$
        - $a_{t+1}$ = $\pi(s_{t+1})$
        - Q($s_t$, $a_t$) += $\alpha$ * ($r_{t+1}$ + $\gamma$ * Q($s_{t+1}$, $a_{t+1}$) - Q($s_t$, $a_t$))
        - $s_t$ = $s_{t+1}$
        - $a_t$ = $a_{t+1}$
    - until $s_t$ == $s_T$

## off-policy: Q-learning
The target policy $\pi$ of Q-learning is greedy, and the behavior policy $b$ can be &epsilon;-greedy or &epsilon;-soft. The update rule is

$${Q_{n+1}(s_t, a_t) = Q_n(s_t, a_t) + \alpha \cdot [r_{t+1} + \gamma \cdot \max_a Q_{n}(s_{t+1}, a) - Q_n(s_t, a_t)], \alpha \in (0, 1]}$$

1. Initialize
    - fill Q($s_t$, $a_t$) with arbitrary real number
    - Q($s_T$, -) = 0
2. Iterate
    - generate the initial state $s_0$ of an episode
    - $s_t$ = $s_0$
    - repeat
        - $a_t$ = $b(s_t)$
        - take action $a_t$
        - observe $r_{t+1}$, $s_{t+1}$
        - Q($s_t$, $a_t$) += $\alpha$ * ($r_{t+1}$ + $\gamma$ * $\max_a Q(s_{t+1}, a)$ - Q($s_t$, $a_t$))
        - $s_t$ = $s_{t+1}$
    - until $s_t$ == $s_T$

## on/off-policy: Expected Sarsa
The target policy $\pi$ of Expected Sarsa is also greedy, but it uses the expected value of $Q_n(s_{t+1}, a)$ for policy evaluation. Therefore, 
the behavior policy $b$ can be &epsilon;-greedy or &epsilon;-soft, or even the same as the target policy. The update rule is

$${Q_{n+1}(s_t, a_t) = Q_n(s_t, a_t) + \alpha \cdot [r_{t+1} + \gamma \cdot \sum_a \pi(a \vert s_{t+1}) Q_{n}(s_{t+1}, a) - Q_n(s_t, a_t)], \alpha \in (0, 1]}$$

1. Initialize
    - fill Q($s_t$, $a_t$) with arbitrary real number
    - Q($s_T$, -) = 0
2. Iterate
    - generate the initial state $s_0$ of an episode
    - $s_t$ = $s_0$
    - repeat
        - $a_t$ = $b(s_t)$ or $\arg \max_a \pi(a \vert s_t)$
        - take action $a_t$
        - observe $r_{t+1}$, $s_{t+1}$
        - Q($s_t$, $a_t$) += $\alpha$ * ($r_{t+1}$ + $\gamma$ * $\sum_a \pi(a \vert s_{t+1}) Q(s_{t+1}, a)$ - Q($s_t$, $a_t$))
        - $s_t$ = $s_{t+1}$
    - until $s_t$ == $s_T$

## off-policy: Double Q-learning
To prevent maximization bias, Double Q-learning uses two independent copies of estimation of the action-value function, $Q_1(s, a)$ and 
$Q_2(s, a)$. Its behavior policy consider the sum or average of $Q_1(s, a)$ and $Q_2(s, a)$, and it can be &epsilon;-greedy or &epsilon;-soft. 
There are two symmetrical update rules, one for $Q_1(s, a)$ and the other for $Q_2(s, a)$. At each time step, Double Q-learning selects one 
from them randomly. These update rules are

$${\begin{align*}
  Q_{1,n+1}(s_t, a_t) &= Q_{1,n}(s_t, a_t) + \alpha \cdot [r_{t+1} + \gamma \cdot Q_{2,n}(s_{t+1}, \arg \max_a Q_{1,n}(s_{t+1}, a)) - Q_{1,n}(s_t, a_t)], \\
  Q_{2,n+1}(s_t, a_t) &= Q_{2,n}(s_t, a_t) + \alpha \cdot [r_{t+1} + \gamma \cdot Q_{1,n}(s_{t+1}, \arg \max_a Q_{2,n}(s_{t+1}, a)) - Q_{2,n}(s_t, a_t)], \\
  &\text{where } \alpha \in (0, 1]
\end{align*}}$$

$${\begin{align*}
  b(s_t) = \begin{cases}
    &\text{$\epsilon$-soft: } a_t \thicksim b(a_t \vert s_t) = (1 - \epsilon) \cdot softmax(s_t, a) + \frac{\epsilon}{\vert A(s_t) \vert}, \\
    &\text{where } softmax(s_t, a) = \frac{e^{Q_1(s_t, a)+Q_2(s_t, a)}}{\sum_b e^{Q_1(s_t, b)+Q_2(s_t, b)}}, b \in A(s_t), \\
    &\text{$\epsilon$-greedy: } a_t \thicksim b(a_t \vert s_t) = \begin{cases}
      1 - \epsilon + \frac{\epsilon}{\vert A(s_t) \vert}, &\text{if }a_t = \arg \max_a Q_1(s_t, a)+Q_2(s_t, a) \\
      \frac{\epsilon}{\vert A(s_t) \vert}, &\text{if }a_t \neq \arg \max_a Q_1(s_t, a)+Q_2(s_t, a)
    \end{cases}
  \end{cases}
\end{align*}}$$

1. Initialize
    - fill $Q_1(s_t, a_t), Q_2(s_t, a_t)$ with arbitrary real number
    - $Q_1(s_T, -)$ = 0
    - $Q_2(s_T, -)$ = 0
2. Iterate
    - generate the initial state $s_0$ of an episode
    - $s_t$ = $s_0$
    - repeat
        - $a_t$ = $b(s_t)$ (&epsilon;-greedy or &epsilon;-soft in $Q_1 + Q_2$)
        - take action $a_t$
        - observe $r_{t+1}$, $s_{t+1}$
        - with 0.5 probability:
            - $Q_1(s_t, a_t) += \alpha \cdot [r_{t+1} + \gamma \cdot Q_2(s_{t+1}, \arg \max_a Q_1(s_{t+1}, a)) - Q_1(s_t, a_t)]$
        - else
            - $Q_2(s_t, a_t) += \alpha \cdot [r_{t+1} + \gamma \cdot Q_1(s_{t+1}, \arg \max_a Q_2(s_{t+1}, a)) - Q_2(s_t, a_t)]$
        - $s_t$ = $s_{t+1}$
    - until $s_t$ == $s_T$

# n-step Temporal Difference Learning
The main difference among MC, one-step TD and n-step TD is the way to approximate the expected return of equation (4), 
$E[G_{t+1} \vert S_t = s_t, A_t = a_t]$. MC uses the sum of sampled discounted rewards up to the terminal state in episodes, $G_{t+1}$. One-step 
TD uses the previous estimation of action-value function $Q_t(s_{t+1}, a)$, either $Q_{1,t}(s_{t+1}, a)$ or $Q_{2,t}(s_{t+1}, a)$. MC's 
approximation is fully sampling, and one-step TD's is fully estimating. Both are at extreme ends of the spectrum of approximation. N-step TD is 
somewhere in the middle, with the sum of (n-1) sampled discounted rewards plus a discounted estimation of action-value at (n-1)-step later time, 
$R_{t+2} + \gamma R_{t+3} + ... + \gamma^{n-2} R_{t+n} + \gamma^{n-1} Q_{t+n-1}(s_{t+n}, a)$. When $n=1$, n-step TD is exactly one-step 
TD; when $t+n \ge T$, n-step TD is the same as MC. MC has no bootstrapping feature, one-step TD has one-step bootstrapping, and n-step TD has 
n-step bootstrapping.

$${\begin{align*}
  G_t &\doteq \sum_{k=0}^{T-t-1} \gamma^k R_{k+t+1} \\
  &= R_{t+1} + \gamma \cdot G_{t+1} &\text{(from (1))} \\
  &= R_{t+1} + \gamma \cdot \sum_{k=1}^{T-t-1} \gamma^{k-1} R_{k+t+1} \\
  \longrightarrow \\
  E[G_{t+1} \vert S_t = s_t, A_t = a_t] &= E[\sum_{k=1}^{T-t-1} \gamma^{k-1} R_{k+t+1} \vert S_t = s_t, A_t = a_t] \\
  &\approxeq \sum_{k=1}^{T-t-1} \gamma^{k-1} r_{k+t+1}, \\
  &\text{where } r_{k+1} \backsim \Pr_{a_k \backsim \pi(a_k \vert s_k)}(r_{k+1} \vert s_k, a_k) &\text{(MC)} \\
  &\approxeq [\sum_a \pi(a \vert s_{t+1}) \cdot Q_t(s_{t+1}, a) \vert \max_a Q_t(s_{t+1}, a) \vert Q_t^{(\prime)}(s_{t+1}, a_{t+1})], \\
  &\text{where } a_{t+1} \backsim \pi(a_{t+1} \vert s_{t+1}) \text{ or } a_{t+1} = \arg \max_a Q_t(s_{t+1}, a) &\text{(one-step TD)} \\
  &\approxeq \sum_{k=1}^{n-1} \gamma^{k-1} r_{k+t+1} + \gamma^{n-1} \cdot E[G_{t+n} \vert S_{t+n-1} = s_{t+n-1}, A_{t+n-1} = a_{t+n-1}], &(8) \\
  &\text{where } r_{k+1} \backsim \Pr_{a_k \backsim \pi(a_k \vert s_k)}(r_{k+1} \vert s_k, a_k) \text{(n-step TD)}
\end{align*}}$$

We define an additional notation to simplify equation (8)

$${\begin{align*}
  G_{t:t+n} &\doteq \sum_{k=0}^{n-1} \gamma^{k} R_{k+t+1} + \gamma^n \cdot Q_{t+n-1}(s_{t+n}, a_{t+n}) \\
  &= R_{t+1} + \sum_{k=1}^{n-1} \gamma^{k} R_{k+t+1} + \gamma^n \cdot Q_{t+n-1}(s_{t+n}, a_{t+n}) \\
  &= R_{t+1} + \gamma \cdot [\sum_{k=1}^{n-1} \gamma^{k-1} R_{k+t+1} + \gamma^{n-1} \cdot Q_{t+n-1}(s_{t+n}, a_{t+n})] \\
  &= R_{t+1} + \gamma \cdot [\sum_{k=0}^{n-2} \gamma^{k} R_{k+t+2} + \gamma^{n-1} \cdot Q_{t+n-1}(s_{t+n}, a_{t+n})] \\
  &\text{where } Q_{t+n-1}(s_{t+n}, a_{t+n}) \approxeq E[G_{t+n} \vert S_{t+n-1} = s_{t+n-1}, A_{t+n-1} = a_{t+n-1}], n \ge 1, 0 \le t \lt T-n \\
  G_{t:t+n} &\doteq G_t, \text{ if } t+n \ge T \\
  \longrightarrow \\
  G_{t+1:t+n} &\doteq \sum_{k=0}^{n-2} \gamma^{k} R_{k+t+2} + \gamma^{n-1} \cdot Q_{t+n-1}(s_{t+n}, a_{t+n}) \\
  &= \sum_{k=1}^{n-1} \gamma^{k-1} R_{k+t+1} + \gamma^{n-1} \cdot Q_{t+n-1}(s_{t+n}, a_{t+n}), \\
  G_{t:t+n} &= R_{t+1} + \gamma \cdot G_{t+1:t+n} \\
  G_{t+1:t+n} &= R_{t+2} + \gamma \cdot G_{t+2:t+n} \\
  G_{t:t+n} &= R_{t+1} + \gamma R_{t+2} + \gamma^2 \cdot G_{t+2:t+n} \\
  &= \sum_{k=0}^{n-1} \gamma^{k} R_{k+t+1} + \gamma^n \cdot G_{t+n:t+n} &(9)
\end{align*}}$$

Generally, the update rule of n-step TD methods are

$${\begin{align*}
  Q_{t+n}(s_t, a_t) &\doteq Q_{t+n-1}(s_t, a_t) + \alpha \cdot [G_{t:t+n} - Q_{t+n-1}(s_t, a_t)] \\
\end{align*}}$$

But different algorithms use different ways to approximate $G_{t+n:t+n}$ of equation (9).

## on-policy: n-step Sarsa
The policy can be &epsilon;-greedy or &epsilon;-soft. The update rule of $G_{t+n:t+n}$ is

$${\begin{align*}
  G_{t+n:t+n} &= Q_{t+n-1}(s_{t+1}, a_{t+1}), \text{ where } a_{t+1} = \pi(s_{t+1}) \\
  \pi(s_{t+1}) &= \begin{cases}
    &\text{$\epsilon$-soft: } a_t \thicksim \pi(a_{t+1} \vert s_{t+1}) = (1 - \epsilon) \cdot softmax(s_{t+1}, a) + \frac{\epsilon}{\vert A(s_t) \vert}, \\
    &\text{where } softmax(s_{t+1}, a) = \frac{e^{Q_{t+n-1(s_{t+1}, a)}}}{\sum_b e^{Q_{t+n-1}(s_{t+1}, b)}}, b \in A(s_{t+1}), \\
    &\text{$\epsilon$-greedy: } a_t \thicksim \pi(a_{t+1} \vert s_{t+1}) = \begin{cases}
      1 - \epsilon + \frac{\epsilon}{\vert A(s_t) \vert}, &\text{if }a_t = \arg \max_a Q_{t+n-1}(s_{t+1}, a) \\
      \frac{\epsilon}{\vert A(s_{t+1}) \vert}, &\text{if }a_{t+1} \neq \arg \max_a Q_{t+n-1}(s_{t+1}, a)
    \end{cases}
  \end{cases}
\end{align*}}$$

1. Initialize
    - fill $Q(s_t, a_t)$ with arbitrary real number.
    - initialize $\pi(a_t \vert s_t)$ following the update rule with respect to $Q(S_t, a_t)$
    - $s_t$ = array[0:n+1]
    - $a_t$ = array[0:n+1]
    - $r_{t+1}$ = array[0:n+1]
2. Iterate
    - generate the initial state $s_0$ of an episode
    - $s_t[0] = s_0$
    - $a_t[0] = \pi(s_t)$
    - T = $\infty$
    - t = 0
    - repeat
        - if t < T:
            - take action $a_t[t \bmod (n+1)]$
            - observe $s_{t+1}, r_{t+1}$
            - $s_t[(t+1) \bmod (n+1)] = s_{t+1}$
            - $r_{t+1}[(t+1) \bmod (n+1)] = r_{t+1}$
            - if $s_{t+1} == s_T$:
                - T = t + 1
            - else:
                - $a_t[(t+1) \bmod (n+1)] = \pi(s_{t+1})$
        - if t >= n-1:
            - if t < T - 1:
                - $G = \sum_{k=(t+1)-(n-1)}^{t+1} \gamma^{k-(t+1)+(n-1)} r_{t+1}[k \bmod (n+1)] + \gamma^n \cdot Q(s_t[(t+1) \bmod (n+1)], a_t[(t+1) \bmod (n+1)])$
            - else:
                - $G = \sum_{k=(t+1)-(n-1)}^T \gamma^{k-(t+1)+(n-1)} r_{t+1}[k \bmod (n+1)]$
            - $Q(s_t[(t+1-n) \bmod (n+1)], a_t[(t+1-n) \bmod (n+1)]) += \alpha \cdot [G - Q(s_t[(t+1-n) \bmod (n+1)], a_t[(t+1-n) \bmod (n+1)])]$
            - update $\pi(a_t \vert s_t[(t+1-n) \bmod (n+1)])$ following the update rule with respect to $Q(S_t, a_t)$
        - t = t + 1
    - until t > T + n - 2

## on-policy: n-step Expected Sarsa
The policy can be &epsilon;-greedy or &epsilon;-soft. The update rule of $G_{t+n:t+n}$ is

$${\begin{align*}
  G_{t+n:t+n} &= \sum_a \pi(a \vert s_{t+1}) Q_{t+n-1}(s_{t+1}, a), \\
  \pi(s_{t+1}) &= \begin{cases}
    &\text{$\epsilon$-soft: } a_t \thicksim \pi(a_{t+1} \vert s_{t+1}) = (1 - \epsilon) \cdot softmax(s_{t+1}, a) + \frac{\epsilon}{\vert A(s_t) \vert}, \\
    &\text{where } softmax(s_{t+1}, a) = \frac{Q_{t+n-1}(s_{t+1}, a)}{\sum_b Q_{t+n-1}(s_{t+1}, b)}, b \in A(s_{t+1}), \\
    &\text{$\epsilon$-greedy: } a_t \thicksim \pi(a_{t+1} \vert s_{t+1}) = \begin{cases}
      1 - \epsilon + \frac{\epsilon}{\vert A(s_t) \vert}, &\text{if }a_t = \arg \max_a Q_{t+n-1}(s_{t+1}, a) \\
      \frac{\epsilon}{\vert A(s_{t+1}) \vert}, &\text{if }a_{t+1} \neq \arg \max_a Q_{t+n-1}(s_{t+1}, a)
    \end{cases}
  \end{cases}
\end{align*}}$$

1. Initialize
    - fill $Q(s_t, a_t)$ with arbitrary real number.
    - initialize $\pi(a_t \vert s_t)$ following the update rule with respect to $Q(S_t, a_t)$
    - $s_t$ = array[0:n+1]
    - $a_t$ = array[0:n+1]
    - $r_{t+1}$ = array[0:n+1]
2. Iterate
    - generate the initial state $s_0$ of an episode
    - $s_t[0] = s_0$
    - $a_t[0] = \pi(s_t)$
    - T = $\infty$
    - t = 0
    - repeat
        - if t < T:
            - take action $a_t[t \bmod (n+1)]$
            - observe $s_{t+1}, r_{t+1}$
            - $s_t[(t+1) \bmod (n+1)] = s_{t+1}$
            - $r_{t+1}[(t+1) \bmod (n+1)] = r_{t+1}$
            - if $s_{t+1} == s_T$:
                - T = t + 1
            - else:
                - $a_t[(t+1) \bmod (n+1)] = \pi(s_{t+1})$
        - if t >= n-1:
            - if t < T - 1:
                - $G = \sum_{k=(t+1)-(n-1)}^{t+1} \gamma^{k-(t+1)+(n-1)} r_{t+1}[k \bmod (n+1)] + \gamma^n \cdot \sum_a \pi(a \vert s_{t+1}) Q(s_t[(t+1) \bmod (n+1)], a_t[(t+1) \bmod (n+1)])$
            - else:
                - $G = \sum_{k=(t+1)-(n-1)}^T \gamma^{k-(t+1)+(n-1)} r_{t+1}[k \bmod (n+1)]$
            - $Q(s_t[(t+1-n) \bmod (n+1)], a_t[(t+1-n) \bmod (n+1)]) += \alpha \cdot [G - Q(s_t[(t+1-n) \bmod (n+1)], a_t[(t+1-n) \bmod (n+1)])]$
            - update $\pi(a_t \vert s_t[(t+1-n) \bmod (n+1)])$ following the update rule with respect to $Q(S_t, a_t)$
        - t = t + 1
    - until t > T + n - 2

## off-policy: n-step Sarsa
The update rule of a off-policy method may need to include the importance sampling ratio. The starting time period and the ending time period 
depends on how many actions are sampled from the behavior policy included in the update rule. For instance, the update rule of n-step Sara for 
value function needs $\rho_{t:t+n-1}$ because $a_t$, ..., $a_{t+n-1}$ are all sampled from $b(a \vert s)$ and influence the estimation of 

$${V_{t+n}(s_t) \doteq V_{t+n-1}(s_t) + \alpha \cdot \rho_{t:t+n-1} \cdot [G_{t:t+n} - V_{t+n-1}(s_t)]}$$

However, the update rule of n-step Sara for action-value function needs $\rho_{t+1:t+n}$ because $a_t$ is certain. For this reason, the update 
rule is

$${Q_{t+n}(s_t, a_t) \doteq Q_{t+n-1}(s_t, a_t) + \alpha \cdot \rho_{t+1:t+n} \cdot [G_{t:t+n} - Q_{t+n-1}(s_t, a_t)]}$$

If the target policy is different from the behavior policy, and the target policy is greedy, this off-policy n-step Expected Sarsa is almost 
the same as n-step Q-learning except the ending time period of the importance sampling ratio $\rho_{t+1:t+n-1}$, one less term than off-policy 
n-step Sara. In this case, $a_{t+n}$ sampled from $b(a \vert s)$ is not used in its update rule but the greedy action, like

$${\begin{align*}
  Q_{t+n}(s_t, a_t) &\doteq Q_{t+n-1}(s_t, a_t) + \alpha \cdot \rho_{t+1:t+n-1} \cdot [G_{t:t+n} - Q_{t+n-1}(s_t, a_t)], \\
  &= Q_{t+n-1}(s_t, a_t) + \alpha \cdot \rho_{t+1:t+n-1} \cdot [\sum_{k=0}^{n-1} \gamma^{k} R_{k+t+1} + \gamma^n \cdot G_{t+n:t+n} - Q_{t+n-1}(s_t, a_t)] \\
  &\text{where } G_{t+n:t+n} = \arg \max_a Q_{t+n-1}(s_{t+n}, a)
\end{align*}}$$

For all the above, the importance sampling ratio is

$${\begin{align*}
  \rho_{t+1:h} &\doteq \prod_{k=t+1}^{min(h, T-1)} \frac{\pi(a_k \vert s_k)}{b(a_k \vert s_k)}, \\
  &\text{where } h = \begin{cases} t+n \text{, for n-step Sarsa} \\
  t+n-1 \text{, for others that $a_{t+n} \backsim b(a_{t+n} \vert s_{t+n})$ is not used} \end{cases}
\end{align*}}$$

Finally, we can generalize n-step TD/MC methods, where $n \in [1, T]$, as

1. Initialize
    - fill $Q(s_t, a_t)$ with arbitrary real number.
    - initialize $\pi(a_t \vert s_t)$, $b(a_t \vert s_t)$ following the update rule with respect to $Q(S_t, a_t)$
    - $s_t$ = array[0:n+1]
    - $a_t$ = array[0:n+1]
    - $r_{t+1}$ = array[0:n+1]
    - algorithms = {'Sarsa': [1, Q(s, a)], 'Q-learning': [0, $\arg \max_a Q(s, a)$], 'Expected Sarsa': = [0, $\sum_a \pi(a \vert s) \cdot Q(s, a)$]}
    - initialize off-policy to true or false
    - initialize MC to true or false
    - name = selected algorithm
    - $a_{t+n}Used$ = algorithms[name][0]
    - $G_{t+n:t+n}$ = algorithms[name][1]
2. Iterate
    - generate the initial state $s_0$ of an episode
    - $s_t[0] = s_0$
    - $a_t[0] = b(s_t)$
    - T = $\infty$
    - if MC:
        - n = T
    - t = 0
    - repeat
        - if t < T:
            - take action $a_t[t \bmod (n+1)]$
            - observe $s_{t+1}, r_{t+1}$
            - $s_t[(t+1) \bmod (n+1)] = s_{t+1}$
            - $r_{t+1}[(t+1) \bmod (n+1)] = r_{t+1}$
            - if $s_{t+1} == s_T$:
                - T = t + 1
                - if MC:
                    - n = T
            - else:
                - $a_t[(t+1) \bmod (n+1)] = b(s_{t+1})$
        - if t >= n-1:
            - h = min(t+n-2+$a_{t+n}Used$, T-1)
            - if (t+1)-(n-1) <= h and off-policy:
                - $\rho = \prod_{k=(t+1)-(n-1)}^{h} \frac{\pi(a_t[k \bmod (n+1)] \vert s_t[k \bmod (n+1)])}{b(a_t[k \bmod (n+1)] \vert s_t[k \bmod (n+1)])}$
            - else:
                - $\rho = 1.0$
            - $G = \sum_{k=(t+1)-(n-1)}^{min(t+1, T)} \gamma^{k-(t+1)+(n-1)} r_{t+1}[k \bmod (n+1)]$
            - if t < T - 1:
                - $G += G_{t+n:t+n}(s_t[(t+1) \bmod (n+1)], a_t[(t+1) \bmod (n+1)])$
            - $Q(s_t[(t+1-n) \bmod (n+1)], a_t[(t+1-n) \bmod (n+1)]) += \alpha \cdot \rho \cdot [G - Q(s_t[(t+1-n) \bmod (n+1)], a_t[(t+1-n) \bmod (n+1)])]$
            - update $\pi(a_t \vert s_t[(t+1-n) \bmod (n+1)])$, $b(a_t \vert s_t[(t+1-n) \bmod (n+1)])$ following the update rule with respect to $Q(S_t, a_t)$
        - t = t + 1
    - until t > T + n - 2

## off-policy without importance sampling: n-step Tree Backup
The idea behind Tree Backup is to include estimations for actions not taken, weighted by $\pi(a \vert s)$. For the taken action, say $a_{t+1}$, 
it is estimated by $\pi(a_{t+1} \vert s_{t+1}) \cdot Q^{\prime}(s_{t+1}, a_{t+1})$, which could be estimated recursively in the same way until 
the bottom level of the tree, say the 3rd level for 3-step Tree Backup. All possible actions are taken into account, not only actions sampled 
from the behavior policy, so importance sampling ratios are not used.

The algorithm may not be needed for the further development of deep reinforcement learning algorithms, for this reason, the details are skipped. 

## n-step $Q(\sigma)$
The idea behind $Q(\sigma)$ is to generalize Tree Backup with other importance sampling based off-policy algorithms. For each taken action, 
a dedicated binary variable $\sigma$ is used to decide which way to estimate its action-value, important sampling or Tree Backup. 

The algorithm may not be needed for the further development of deep reinforcement learning algorithms, for this reason, the details are skipped. 

# Reference
- Carnegie Mellon University, Fragkiadaki, Katerina, et al. 2024. "10-403 Deep Reinforcement Learning" As of 8 November, 2024. 
https://cmudeeprl.github.io/403website_s24/.
- Sutton, Richard S., and Barto, Andrew G. 2018. Reinforcement Learning - An indroduction, second edition. The MIT Press.
- Towers, et al. 2024. "Gymnasium: A Standard Interface for Reinforcement Learning Environments", [arXiv:2407.17032](https://arxiv.org/abs/2407.17032).
- Farama Foundation, n.d. Gymnasium. https://github.com/farama-Foundation/gymnasium?tab=readme-ov-file
