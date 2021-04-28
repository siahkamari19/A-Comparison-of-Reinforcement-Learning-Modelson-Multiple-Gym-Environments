import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
from collections import deque
import argparse


class SAC:
	def __init__(self, env_name, path=None):
		seed = 41
		self.env = gym.make(env_name)
		self.env.seed(seed)
		self.gamma = 0.99
		self.max_steps_per_episode = 10000
		self.eps = np.finfo(np.float32).eps.item()

		self.num_inputs = self.env.observation_space.shape[0]
		self.num_actions = self.env.action_space.n
		self.num_hidden = 128

		self.model = self._create_model(path)

		self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
		self.huber_loss = keras.losses.Huber()

	def _create_model(self, path):
		inputs = layers.Input(shape=(self.num_inputs,))
		common = layers.Dense(self.num_hidden, activation="relu")(inputs)
		action = layers.Dense(self.num_actions, activation="softmax")(common)
		critic = layers.Dense(1)(common)

		model = keras.Model(inputs=inputs, outputs=[action, critic])
		if path is not None:
			model.load_weights(path)
		return model

	def test(self, path, render=True):
		#self.model.load_weights(path)
		action_probs_history = []
		critic_value_history = []
		rewards_history = []
		running_reward = 0
		episode_count = 0
		state = self.env.reset()
		episode_reward = 0.0
		with tf.GradientTape() as tape:
			for timestep in range(1, self.max_steps_per_episode):
		 		if render:
		 			self.env.render()
		 		state = tf.convert_to_tensor(state)
		 		state = tf.expand_dims(state, 0)
		 		action_probs, critic_value = self.model(state)
		 		critic_value_history.append(critic_value[0, 0])
		 		action = np.argmax(action_probs)
		 		state, reward, done, _ = self.env.step(action)
		 		rewards_history.append(reward)
		 		episode_reward += reward
		 		if done:
			 		break
		return episode_reward

	def train(self):
		action_probs_history = []
		critic_value_history = []
		rewards_history = []
		running_reward = 0
		episode_count = 0

		average = deque(maxlen=200)

		while True:  # Run until solved
		    state = self.env.reset()
		    episode_reward = 0
		    with tf.GradientTape() as tape:

		        for timestep in range(1, self.max_steps_per_episode):
		            #env.render()  # Adding this line would show the attempts
		            # of the agent in a pop up window.

		            state = tf.convert_to_tensor(state)
		            state = tf.expand_dims(state, 0)

		            # Predict action probabilities and estimated future rewards
		            # from environment state
		            action_probs, critic_value = self.model(state)
		            critic_value_history.append(critic_value[0, 0])

		            # Sample action from action probability distribution
		            action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
		            action_probs_history.append(tf.math.log(action_probs[0, action]))

		            # Apply the sampled action in our environment
		            state, reward, done, _ = self.env.step(action)
		            rewards_history.append(reward)
		            episode_reward += reward

		            if done:
		                break

		        # Update running reward to check condition for solving
		        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

		        # Calculate expected value from rewards
		        # - At each timestep what was the total reward received after that timestep
		        # - Rewards in the past are discounted by multiplying them with gamma
		        # - These are the labels for our critic
		        returns = []
		        discounted_sum = 0
		        for r in rewards_history[::-1]:
		            discounted_sum = r + self.gamma * discounted_sum
		            returns.insert(0, discounted_sum)

		        # Normalize
		        returns = np.array(returns)
		        returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
		        returns = returns.tolist()

		        # Calculating loss values to update our network
		        history = zip(action_probs_history, critic_value_history, returns)
		        actor_losses = []
		        critic_losses = []
		        for log_prob, value, ret in history:
		            # At this point in history, the critic estimated that we would get a
		            # total reward = `value` in the future. We took an action with log probability
		            # of `log_prob` and ended up recieving a total reward = `ret`.
		            # The actor must be updated so that it predicts an action that leads to
		            # high rewards (compared to critic's estimate) with high probability.
		            diff = ret - value
		            actor_losses.append(-log_prob * diff)  # actor loss

		            # The critic must be updated so that it predicts a better estimate of
		            # the future rewards.
		            critic_losses.append(
		                self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
		            )

		        # Backpropagation
		        loss_value = sum(actor_losses) + sum(critic_losses)
		        grads = tape.gradient(loss_value, self.model.trainable_variables)
		        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

		        # Clear the loss and reward history
		        action_probs_history.clear()
		        critic_value_history.clear()
		        rewards_history.clear()
		        average.append(episode_reward)

		    # Log details
		    episode_count += 1
		    if episode_count % 10 == 0:
		        template = "running reward: {:.2f} at episode {}"
		        print(template.format(running_reward, episode_count))
		        model.save_weights('SAC_CartPole.h5')

		    j = list(average)
		    x = np.mean(j[-50:])
		    if x > -100:  # Condition to consider the task solved
		        print("Solved at episode {}!".format(episode_count))
		        return self.model
		        break


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--game", type=str, help="Environment Name", default='CartPole-v1')
	parser.add_argument("--mode", type=str, help="train - train model, test - test model", default='train')
	parser.add_argument("--path", type=str, help="policy path", default=None)
	return parser

if __name__ == '__main__':
	args = get_args().parse_args()
	sac = SAC(args.game, args.path)

	if args.mode == 'train':
		sac.train()
	else:
		total = 0
		no = 100
		for i in range(no):
			val = sac.test(args.game)
			print('{} episode: {}'.format(i, val))
			total += 1
		print('AVG SCORE FOR 100 EPISODES', val / no)	