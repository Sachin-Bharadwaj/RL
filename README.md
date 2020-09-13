# Dueling Double DQN agent playing Super Mario

**Start of Training**
![Alt Text](https://github.com/Sachin-Bharadwaj/RL/blob/master/Mario/recording-initial/ezgif.com-gif-maker.gif)

**Trained Agent playing Super Mario**
![Alt Text](https://github.com/Sachin-Bharadwaj/RL/blob/master/Mario/recording-trained/ezgif.com-gif-maker.gif)

**Last 100 epsideo average reward distribution, snapshot taken every 50 episodes**
![Alt Text](https://github.com/Sachin-Bharadwaj/RL/blob/master/Mario/recording-trained/last100Reward_distribution.png)
As you can observe as training proceeds, the reward distribution tail becomes thicker indicating that agent is able to reach goal more often

**PI_VI_FrozenLake 8x8 stochastic environment**
- MDP Known
  - Implemented Policy Iteration (Policy Evaluation and Policy Improvement)
  - Implemented Value Iteration
- Prediction Problem (MPD not Known)
  - Monte Carlo (First visit/Every visit)
  - TD(0), TD(n), TD(lambda)
- Control Problem (MDP not Known)
  - Generalized Policy Iteration using (MonteCarlo, SARSA)
  - Q-learning
  - Double Q-learning


