import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import time
from resnet import ResNet18

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet_path = "cifar100_resnet18.pt"
resnet = ResNet18(100).to(device)
resnet.load_state_dict(torch.load(resnet_path))
for param in resnet.parameters():
    param.requires_grad = False
no_pool_resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor(nn.Module):
    def __init__(self, proprio_dim, action_dim):
        super(Actor, self).__init__()

        # self.fc1_1 = nn.Linear(8192, 2048)
        # self.fc1_2 = nn.Linear(2048, 512)
        # self.fc2_1 = nn.Linear(8192, 2048)
        # self.fc2_2 = nn.Linear(2048, 512)

        self.tanh = nn.Tanh()
        self.fc3 = nn.Linear(proprio_dim, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, action_dim)

    def forward(self, state):
        # i_1 = no_pool_resnet(state[0].to(device)).view(state[0].size(0),-1)
        # i_1 = self.tanh(self.fc1_1(i_1))
        # i_1 = self.tanh(self.fc1_2(i_1))
        # i_2 = no_pool_resnet(state[1].to(device)).view(state[1].size(0),-1)
        # i_2 = self.tanh(self.fc2_1(i_2))
        # i_2 = self.tanh(self.fc2_2(i_2))
        # s = torch.cat([i_1, i_2, state[2].view(state[2].size(0), -1).to(device)], dim=1)
        s = self.tanh(self.fc3(state[0].to(device)))
        s = self.tanh(self.fc4(s))
        return self.tanh(self.fc5(s))

class Critic(nn.Module):
    def __init__(self, proprio_dim):
        super(Critic, self).__init__()
        # self.fc1_1 = nn.Linear(8192, 2048)
        # self.fc1_2 = nn.Linear(2048, 512)
        # self.fc2_1 = nn.Linear(8192, 2048)
        # self.fc2_2 = nn.Linear(2048, 512)

        self.tanh = nn.Tanh()
        self.fc3 = nn.Linear(proprio_dim, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, state):
        # i_1 = no_pool_resnet(state[0].to(device)).view(state[0].size(0),-1)
        # i_1 = self.tanh(self.fc1_1(i_1))
        # i_1 = self.tanh(self.fc1_2(i_1))
        # i_2 = no_pool_resnet(state[1].to(device)).view(state[1].size(0),-1)
        # i_2 = self.tanh(self.fc2_1(i_2))
        # i_2 = self.tanh(self.fc2_2(i_2))
        # s = torch.cat([i_1, i_2, state[2].view(state[2].size(0), -1).to(device)], dim=1)
        s = self.tanh(self.fc3(state[0].to(device)))
        s = self.tanh(self.fc4(s))
        return self.fc5(s)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = preprocess(state)
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_states = [torch.cat(i, dim=0).detach() for i in list(zip(*memory.states))]
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def preprocess(inputs):
    """Preprocess images and proprio input"""
    # image_tensor = torch.FloatTensor(inputs[0][0])
    # image_tensor = image_tensor.permute([2, 0, 1])
    # image_batch1 = image_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    #
    # image_tensor = torch.FloatTensor(inputs[1][0])
    # image_tensor = image_tensor.permute([2, 0, 1])
    # image_batch2 = image_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    proprio_tensor = torch.FloatTensor(inputs[0][0].reshape(1, -1))
    proprio_batch = proprio_tensor.unsqueeze(0)
    return [proprio_batch]


def main():
    ############## Hyperparameters ##############
    env_name = "MouseAgent-v2"
    solved_reward = 70         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 3000         # max training episodes
    max_timesteps = 600        # max timesteps in one episode

    update_timestep = 2400      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    # creating environment
    # channel = EngineConfigurationChannel()
    # env = UnityEnvironment(file_name='mouse_env/linux_exec', side_channels=[channel])
    # channel.set_configuration_parameters(time_scale = 20.0)
    env = UnityEnvironment(base_port=5004)
    env.reset()

    group_name = env.get_behavior_names()[0]
    group_spec = env.get_behavior_spec(group_name)
    step_result = env.get_steps(group_name)
    print("Number of observations : ", group_spec.observation_shapes)

    action_dim = 3
    state_dim = 84

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes+1):

        state = step_result[0].obs
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            env.set_actions(group_name, np.array([np.clip(action, -1, 1)]))

            env.step()
            step_result = env.get_steps(group_name)

            state = step_result[0].obs
            reward = step_result[0].reward[0]

            done = False
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            # if render:
            #     env.render()
            # if done:
            #     break
        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
        env.reset()
        env.step()
        step_result = env.get_steps(group_name)
if __name__ == '__main__':
    main()
