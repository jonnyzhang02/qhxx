import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('位置范围 = {}'.format((env.unwrapped.min_position,env.unwrapped.max_position)))
print('速度范围 = {}'.format((-env.unwrapped.max_speed,env.unwrapped.max_speed)))
print('目标位置 = {}'.format(env.unwrapped.goal_position))

# # 如果单纯一直向右加速，是到不了的
# positions, velocities = [], []
# observation, _ = env.reset()
# while True:
# 	positions.append(observation[0])
# 	velocities.append(observation[1])
# 	# print("位置 = {:.3f}\t速度 = {:.3f}\n".format(observation[0], observation[1]))
# 	next_observation, reward, terminated, truncated , __ = env.step(2)
# 	if terminated or truncated:
# 		break
# 	observation = next_observation

# if next_observation[0] > 0.5:
# 	print('成功到达')
# else:
# 	print('失败退出')

# # 绘制位置和速度图像
# fig, ax = plt.subplots()
# ax.plot(positions, label='position')
# ax.plot(velocities, label='velocity')
# ax.legend()
# plt.show()

class DQNReplayer:
	def __init__(self, capacity):
		self.memory = pd.DataFrame(index=range(capacity),
				columns=['observation', 'action', 'reward',
				'next_observation', 'done'])
		self.i = 0
		self.count = 0
		self.capacity = capacity
	
	def store(self, *args):
		self.memory.loc[self.i] = args
		self.i = (self.i + 1) % self.capacity
		self.count = min(self.count + 1, self.capacity)
		
	def sample(self, size):
		indices = np.random.choice(self.count, size=size)
		return (np.stack(self.memory.loc[indices, field]) for field in
				self.memory.columns)


class DoubleDQNAgent:
	def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.001,
				replayer_capacity=10000, batch_size=64):
		observation_dim = env.observation_space.shape[0]
		self.action_n = env.action_space.n
		self.gamma = gamma
		self.epsilon = epsilon
		
		self.batch_size = batch_size
		self.replayer = DQNReplayer(replayer_capacity) # 经验回放
		
		self.evaluate_net = self.build_network(input_size=observation_dim,
				output_size=self.action_n, **net_kwargs, hidden_sizes=[64 ,128]) # 评估网络
		self.target_net = self.build_network(input_size=observation_dim,
				output_size=self.action_n, **net_kwargs, hidden_sizes=[64 ,128]) # 目标网络

		self.target_net.set_weights(self.evaluate_net.get_weights())

	def build_network(self, input_size, hidden_sizes, output_size,
				activation=tf.nn.relu, output_activation=None,
				learning_rate=0.01): # 构建网络
		model = keras.Sequential()  
		# 隐藏层
		for layer, hidden_size in enumerate(hidden_sizes): 
			kwargs = dict(input_shape=(input_size,)) if not layer else {} 
			model.add(keras.layers.Dense(units=hidden_size, activation=activation, **kwargs)) 		
		# 输出层
		model.add(keras.layers.Dense(units=output_size,activation=output_activation)) 				
		# 优化器
		optimizer = tf.optimizers.Adam(lr=learning_rate) 	

		model.compile(loss='mse', optimizer=optimizer)      
		return model
		
	def learn(self, observation, action, reward, next_observation, done):
		self.replayer.store(observation, action, reward, next_observation,
				done) # 存储经验
		observations, actions, rewards, next_observations, dones = \
				self.replayer.sample(self.batch_size) # 经验回放
		next_eval_qs = self.evaluate_net.predict(next_observations)
		next_actions = next_eval_qs.argmax(axis=-1)
		next_qs = self.target_net.predict(next_observations)
		next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions] 
		us = rewards + self.gamma * next_max_qs * (1. - dones)
		targets = self.evaluate_net.predict(observations)
		targets[np.arange(us.shape[0]), actions] = us
		self.evaluate_net.fit(observations, targets, verbose=0)

		if done:
			self.target_net.set_weights(self.evaluate_net.get_weights())


	def decide(self, observation): # epsilon贪心策略
		if np.random.rand() < self.epsilon:
			return np.random.randint(self.action_n)
		qs = self.evaluate_net.predict(observation[np.newaxis])
		return np.argmax(qs)

	def save_model(self, filepath):
		self.evaluate_net.save(filepath)
				



def play_qlearning(env, agent, train=False, render=False):
	episode_reward = 0
	observation, _ = env.reset()
	while True:
		if render:
			env.render()
		action = agent.decide(observation)
		next_observation, reward, terminated, truncated , __ = env.step(action)
		episode_reward += reward
		if train:
			agent.learn(observation, action, reward, next_observation,
					terminated or truncated)
		if terminated or truncated:
			break
		observation = next_observation
	return episode_reward


positions = []
velocities = []
env = gym.make('MountainCar-v0')
np.random.seed(0)
agent = DoubleDQNAgent(env)
IS_TRAIN = False
if IS_TRAIN:
	for episode in range(1000):
		episode_reward = play_qlearning(env, agent, train=True)
		print('episode: ', episode, 'episode_reward:', episode_reward)

		if episode_reward > -200:
			agent.save_model(filepath=f'./models/model_at_episode_{episode}_reward_{episode_reward}.h5')
else:
	agent.evaluate_net = keras.models.load_model('./models/model_at_episode_99_reward_-134.0.h5')
	positions, velocities = [], []
	observation, _ = env.reset()
	while True:
		positions.append(observation[0])
		velocities.append(observation[1])
		action = agent.decide(observation)
		print(action)
		# print("位置 = {:.3f}\t速度 = {:.3f}\n".format(observation[0], observation[1]))
		next_observation, reward, terminated, truncated , __ = env.step(action)
		if terminated or truncated:
			break
		observation = next_observation

	if next_observation[0] > 0.5:
		print('成功到达')
	else:
		print('失败退出')

	# 绘制位置和速度图像
	fig, ax = plt.subplots()
	ax.plot(positions, label='position')
	ax.plot(velocities, label='velocity')
	ax.legend()
	plt.show()

env.close()



