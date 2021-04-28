import gym
import torch
from tqdm import tqdm
import random
import numpy as np


from PPO.model import (
    PolicyNetwork,
    ValueNetwork,
    device,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--policy_path")
    parser.add_argument("--env_name")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--max_timesteps", type=int, default=600)
    parser.add_argument("--render", type=bool, default=False)


    state_scale = 1.0

    args = parser.parse_args()

    policy_path = args.policy_path
    env_name = args.env_name

    render = args.render

    n_episodes = args.n_episodes
    max_timesteps = args.max_timesteps


    env = gym.make(env_name)
    env.seed(123)
    observation = env.reset()
    n_actions = env.action_space.n
    feature_dim = observation.size

    policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)

    policy_model.load_state_dict(torch.load(policy_path))

    frames = []
    sum_reward = 0
    for _ in range(n_episodes):
        observation = env.reset()
        curr = 0
        while True:

            if render:
                env.render()
            action = policy_model.best_action(observation / state_scale)

            new_observation, reward, done, info = env.step(action)
            sum_reward += reward
            curr += reward

            if done:
                print(_, curr)
                break

            observation = new_observation
    print('AVG', sum_reward/n_episodes)
