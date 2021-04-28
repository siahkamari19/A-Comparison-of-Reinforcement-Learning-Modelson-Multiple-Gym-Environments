from pathlib import Path
import argparse

import gym
import numpy as np 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from PPO.model import (
    PolicyNetwork,
    ValueNetwork,
    device,
    train_value_network,
    train_policy_network
)

from PPO.replay import Episode, History
from time import time


def configs(env_name):
    if env_name == 'CartPole-v1':
        #return 2, 20, 64
        return 4, 20, 64

    if env_name == 'LunarLander-v2':
        return 4, 50, 128

    if env_name == 'Acrobot-v1':
        return 4, 30, 128


def main(env_name, reward_scale, clip, log_dir, lr, state_scale):
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=env_name, comment=env_name)
    env = gym.make(env_name)
    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

    value_model = ValueNetwork(in_dim=feature_dim).to(device)
    value_optimizer = optim.Adam(value_model.parameters(), lr=lr)

    policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=lr)

    n_epoch, max_episodes, batch_size = configs(env_name)

    history = History()
    episode_ite = 0 
    episode_y = 0

    while True:
        #SAVE CODE

        rewards = []
        for episode_i in range(max_episodes):
            last_observation = env.reset()
            episode = Episode()

            total_reward = 0.0
            done = False
            
            while not done:
                action, log_probability = policy_model.sample_action(last_observation / state_scale)
                value = value_model.state_value(last_observation / state_scale)

                new_observation, reward, done, _ = env.step(action)
                total_reward += reward
                episode.append(
                    observation=last_observation / state_scale,
                    action=action,
                    reward=reward,
                    value=value,
                    log_probability=log_probability,
                    reward_scale=reward_scale
                )
                
                last_observation = new_observation

            episode.end_episode(last_value=0)
            rewards.append(total_reward)
            history.add_episode(episode)
            episode_y += 1
            writer.add_scalar(
                "Average Episode Reward",
                reward_scale * np.sum(episode.rewards),
                episode_y,
            )
            writer.add_scalar(
                "Average Probabilities",
                np.exp(np.mean(episode.log_probabilities)),
                episode_y,
            )

        reward = np.mean(rewards)

        print(episode_ite, reward)

        if reward >= env.spec.reward_threshold:
            check_rewards = []
            done = False
            for i in range(100):
                while not done:
                    action, log_probability = policy_model.sample_action(last_observation / state_scale)
                    value = value_model.state_value(last_observation / state_scale)

                    new_observation, reward, done, _ = env.step(action)
                    total_reward += reward
                    episode.append(
                        observation=last_observation / state_scale,
                        action=action,
                        reward=reward,
                        value=value,
                        log_probability=log_probability,
                        reward_scale=reward_scale
                    )
                    last_observation = new_observation
            
                check_rewards.append(total_reward)
            reward2 = np.mean(check_rewards)
            if reward2 >= env.spec.reward_threshold:
                torch.save(
                    policy_model.state_dict(),
                    Path(log_dir) / (env_name + f"_policy.pth"),
                )
                torch.save(
                    value_model.state_dict(),
                    Path(log_dir) / (env_name + f"_value.pth"),
                )
                print('NUMBER OF EPISODES AFTER TRAIN: {}'.format(episode_ite))
                break

        history.build_dataset()
        data_loader = DataLoader(history, batch_size=batch_size, shuffle=True)

        policy_loss = train_policy_network(
            policy_model, policy_optimizer, data_loader, epochs=n_epoch, clip=clip
        )

        value_loss = train_value_network(
            value_model, value_optimizer, data_loader, epochs=n_epoch
        )

        history.free_memory()
        episode_ite += max_episodes

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--game", type=str, help="Environment Name", default='CartPole-v1')

    return parser

if __name__ == '__main__':
    start = time()
    args = get_args().parse_args()
    env_name = args.game
    main(
        reward_scale=20.0,
        clip=0.2,
        env_name=env_name,
        lr=0.001,
        state_scale=1.0,
        log_dir='logs/'
    )
    print('TOTAL TIME', time() - start)
