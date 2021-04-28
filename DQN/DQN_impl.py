import random
from collections import deque
import argparse

import gym
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

EPISODES = 10000

ENV_SETUP = {
    0: ("CartPole-v1", 497),
    1: ("Acrobot-v1", -100),
    2: ("LunarLander-v2", 200)
}


def setup_summary():
    """ summary operators for tensorboard """
    episode_total_reward = tf.Variable(0.)

    tf.summary.scalar('Total_Reward/Episode', episode_total_reward)

    summary_vars = [episode_total_reward]

    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]

    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]

    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--game", type=int, help="0 for CartPole, 1 for Acrobot, and 2 for Lunar Lander", default=0)
    parser.add_argument("--mode", type=str, help='train- To train the model, test - To test the model', default='train')
    parser.add_argument("--model_path", type=str, help='path of the trained model')

    return parser


class DQNAgent:
    def __init__(self, state_size, action_size, game_mode):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.batch_size = 128

        self.target = self._build_model()
        self.model = self._build_model()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.game_mode = game_mode

        self.stop_criteria = ENV_SETUP.get(game_mode)[1]

        self.summary_placeholders, self.update_ops, self.summary_op = setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/' + self.name, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):

        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target(self):
        return self.target.set_weights(self.model.get_weights())

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @property
    def name(self):

        name = [ENV_SETUP.get(self.game_mode)[0]]

        return ''.join(name[::-1])

    def save_model(self, name=""):
        if len(name) > 0:
            self.model.save_weights(name)
        else:
            self.model.save_weights(self.name)

    def load_model(self, name=""):
        if len(name) > 0:
            self.model.load_weights(name)
        else:
            self.model.load_weights(self.name)
        print("Model Loaded!")

    def act(self, state):
        action_values = self.model.predict([state])[0]

        a_max = np.argmax(action_values)

        if np.random.rand() <= self.epsilon:
            policy = np.ones(self.action_size) * self.epsilon / self.action_size
            policy[a_max] += 1. - self.epsilon
            a = np.random.choice(self.action_size, p=policy)
        else:
            a = a_max
        return a

    def replay(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        # batch data
        states = np.ndarray(shape=(self.batch_size, self.state_size))
        actions = np.ndarray(shape=(self.batch_size, 1))
        rewards = np.ndarray(shape=(self.batch_size, 1))
        next_states = np.ndarray(shape=(self.batch_size, self.state_size))
        done = np.ndarray(shape=(self.batch_size, 1))

        temp = 0
        for exp in mini_batch:
            states[temp] = exp[0]
            actions[temp] = exp[1]
            rewards[temp] = exp[2]
            next_states[temp] = exp[3]
            done[temp] = exp[4]
            temp += 1

        qhat_next = self.target.predict(next_states)

        qhat_next = qhat_next * (np.ones(shape=done.shape) - done)

        qhat_next = np.max(qhat_next, axis=1)

        qhat = self.model.predict(states)

        for i in range(self.batch_size):
            a = actions[i, 0]
            qhat[i, int(a)] = rewards[i] + self.gamma * qhat_next[i]

        q_target = qhat

        self.model.fit(states, q_target, epochs=1, verbose=0)

    def reset_epsilon(self):
        self.epsilon = -1

    def train(self, model_path=None):
        # load model if you are continuing training
        if model_path is not None:
            self.load_model()
        rewards = []
        aver_reward = []
        aver = deque(maxlen=200)
        step = 0
        # for e in range(EPISODES):
        e = 0
        while True:
            e += 1
            # reset state in the beginning of each game
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])

            done = False
            score = 0
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            while not done:
                step += 1

                # turn this on if you want to render
                # env.render()

                action = self.act(state)

                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                agent.add_to_buffer(state, action, reward, next_state, done)

                state = next_state

                score += reward

                if done:
                    if e % 50 == 0:
                        print("episode: {}, score: {}, step: {}"
                              .format(e, score, step))
                    # rest is summary stuff
                    stats = [score]

                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])})

                    summary_str = self.sess.run(self.summary_op)

                    self.summary_writer.add_summary(summary_str, e + 1)

                    break

            # If there is enough then learn
            if len(self.memory) > self.batch_size:
                self.replay()
                self.update_target()

            # rest is exit condition and average reward
            aver.append(score)
            aver_reward.append(np.mean(aver))

            rewards.append(score)
            j = list(aver)
            x = np.mean(j[-50:])
            if x > self.stop_criteria:
                print("Ending Average:", x, "Ending episode:", e)
                break
            if e % 50 == 0:
                print("Average of last 50: " + str(x))

        # save when done
        self.save_model()

    def eval(self, name=""):
        self.load_model(name)
        env.seed(123)
        tests = 100
        tot = 0.0

        # This prevents random actions
        self.reset_epsilon()

        for e in range(tests):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])

            done = False

            score = 0.0
            while not done:
                # turn this on if you want to render
                env.render()

                action = self.act(state)

                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                state = next_state
                score += reward

                if done:
                    print("episode: {}/{}, score: {}"
                          .format(e, tests, score))
                    tot += score

                    break

        print("average: {}".format(tot / tests))


if __name__ == "__main__":
    args = build_parser().parse_args()  # command line start
    env = gym.make(ENV_SETUP.get(args.game)[0])
    obs = env.observation_space.shape[0]
    acts = env.action_space.n
    agent = DQNAgent(obs, acts, args.game)
    if args.mode == 'train':
        agent.train()
        agent.eval()
    else:
        agent.eval(args.model_path)
