import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt

os.chdir('./Continuous')

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu                          # 均值 
        self.theta        = theta                       # 系数
        self.sigma        = max_sigma                   # 标准差
        self.max_sigma    = max_sigma                   # 最大标准差
        self.min_sigma    = min_sigma                   # 最小标准差
        self.decay_period = decay_period                # 衰减周期
        self.action_dim   = action_space.shape[0]       # action的维度
        self.low          = action_space.low            # action的最小值
        self.high         = action_space.high           # action的最大值
        self.reset()
        
    def reset(self):
        """
        重置噪声
        """
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        """
        计算噪声
        """
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        """
        根据当前的action和时间t，计算噪声并返回
        """
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)



class DDPG():
    def __init__(self, state_dim,     
                    action_dim, 
                    hidden_size, 
                    actor_lr, 
                    critic_lr, 
                    discount_factor, 
                    tau, 
                    action_space):
        
        self.state_dim          = state_dim                                     # state的维度
        self.action_dim         = action_dim                                    # action的维度
        self.hidden_size        = hidden_size                                   # 隐藏层的维度
        self.actor_lr           = actor_lr                                      # actor的学习率
        self.critic_lr          = critic_lr                                     # critic的学习率
        self.discount_factor    = discount_factor                               # 折扣因子
        self.tau                = tau                                           # 软更新的参数
        self.noise              = OUNoise(action_space)

        self.actor              = Actor(                                    
                                    self.state_dim, 
                                    self.hidden_size, 
                                    self.action_dim)                            # 创建actor网络
        self.actor_target       = Actor(
                                    self.state_dim, 
                                    self.hidden_size, 
                                    self.action_dim)                            # 创建actor_target网络
        self.actor_optimizer    = optim.Adam(
                                    self.actor.parameters(),                    # actor的优化器
                                    lr=self.actor_lr)

        self.critic             = Critic(
                                    self.state_dim + self.action_dim, 
                                    self.hidden_size)                           # 创建critic网络
        self.critic_target      = Critic(
                                    self.state_dim + self.action_dim,           
                                    self.hidden_size)
        self.critic_optimizer   = optim.Adam(                                   # 创建critic_target网络
                                    self.critic.parameters(), 
                                    lr=self.critic_lr)                          # critic的优化器

        self.memory = []                                                        # 经验池        

    def select_action(self, state):
        state = torch.from_numpy(state).float()             # 将numpy数组转化为PyTorch张量
        action = self.actor(state).detach().numpy()         # 通过actor网络生成动作，并将PyTorch张量转化为numpy数组
        return self.noise.get_action(action)                # 通过噪声处理生成最终的动作

    def update(self, batch_size):
        # 检查经验池是否足够大，能否提供足够的样本进行训练
        if len(self.memory) < batch_size:
            return

        # 从经验池中随机取样
        batch = random.sample(self.memory, batch_size)

        # 3将取样的数据转换成numpy数组的形式
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*batch))

        # 将numpy数组转换成PyTorch张量的形式
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        # 计算目标Q值
        target_q = self.critic_target(torch.cat([next_state_batch, self.actor_target(next_state_batch)], 1))
        target_q = reward_batch + (1 - done_batch) * self.discount_factor * target_q

        # 计算当前的Q值
        q = self.critic(torch.cat([state_batch, action_batch], 1))

        # 更新critic网络，最小化预测的Q值和目标Q值的均方误差
        critic_loss = F.mse_loss(target_q, q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新actor网络，最大化Q值
        actor_loss = -self.critic(torch.cat([state_batch, self.actor(state_batch)], 1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新target网络的参数，让target网络慢慢接近实际网络
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save_model(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
    
    def load_model(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = Actor(self.state_dim, self.hidden_size, self.action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = Critic(self.state_dim + self.action_dim, self.hidden_size)
        self.critic_target.load_state_dict(self.critic.state_dict())




env = gym.make('MountainCarContinuous-v0')
agent = DDPG(state_dim=2, 
                action_dim=1, 
                hidden_size=64, 
                actor_lr=1e-4, 
                critic_lr=1e-3, 
                discount_factor=0.99, 
                tau=1e-2,
                action_space=env.action_space)

IS_TRAIN = False
if IS_TRAIN:
    episodes = 500
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        actions = []
        while 1:
            action = agent.select_action(state)
            actions.append(action.item())
            next_state, reward, terminated, truncated , __ = env.step(action)
            done = terminated or truncated
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            agent.update(32)
            if terminated:
                print(f'第{episode}局游戏成功,reward为{reward}')
                os.makedirs(f'./models/episode{episode}', exist_ok=True)
                agent.save_model(f'./models/episode{episode}/ddpg')
                break
            if truncated:
                print(f'第{episode}局游戏失败,reward为{reward}')
                break
    # if episode % 50 == 0:
    #     print(f'第{episode}局的动作为{actions}')

else:
    # Load the trained model
    agent.load_model("./models/ddpg")

    # 初始化
    state, _ = env.reset()
    positions = []
    velocities = []
    actions = []

    done = False
    while not done:
        action = agent.select_action(state)
        
        # Append to the lists
        positions.append(state[0])
        velocities.append(state[1])
        actions.append(action[0])
        
        state, reward, terminated, truncated , __ = env.step(action)

        done = terminated or truncated

    # Plotting
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15,9))

    # Plot positions
    ax[0].plot(positions)
    ax[0].set_ylabel('Position')

    # Plot velocities
    ax[1].plot(velocities)
    ax[1].set_ylabel('Velocity')

    # Plot actions
    ax[2].plot(actions)
    ax[2].set_ylabel('Action')
    ax[2].set_xlabel('Steps')

    plt.show()





