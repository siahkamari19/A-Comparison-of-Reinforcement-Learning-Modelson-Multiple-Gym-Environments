##Abstract

The goal of this project was to take reinforcement learn-
ing algorithms and apply each of them to multiple domains.
We chose Deep Q-Learning (DQN), Proximal Policy Op-
timization (PPO), and Soft Actor-Critic (SAC) as they are
sufficiently different from one another and have their own
strengths and weaknesses. To compare our chosen algo-
rithms, we used three OpenAI Gym environments: CartPole,
Lunar Lander, and Acrobot. Then, we compared them to each
other to determine which algorithm was the quickest, gener-
ated the best results, and a combination of both. Priority was
given to algorithm which ran for the least episodes before
converging. Each algorithm was run with the same conver-
gence criteria based on the environment to standardize the
results. For each algorithm we determined that certain hyper-
parameters and configurations were not conducive to success,
and found standard models that applied to all of our domains.
In the end we determined that each algorithm was capable
of success, but PPO and SAC were much more efficient in
more complicated domains, with PPO being the fastest over-
all. In our simplest domain, CartPole, we found that DQN
performed similarly to the others with little difference in effi-
ciency or accuracy of results.
