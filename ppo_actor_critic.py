import torch
import torch.nn as nn
from torchvision import transforms
from torch.distributions import MultivariateNormal

from resnet import ResNet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load resnet model and use pre-global maxpool output.
resnet_path = "cifar100_resnet18.pt"
resnet = ResNet18(100).to(device)
resnet.load_state_dict(torch.load(resnet_path))
for param in resnet.parameters():
    param.requires_grad = False
no_pool_resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

# Using normalization from training the cutout resnet model.
transform = transforms.Compose([transforms.Normalize(
mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])


def preprocess(inputs):
    """Preprocess images and proprio input into tensors with correct dimensions"""
    image_tensor = torch.FloatTensor(inputs[0][0])
    image_tensor = image_tensor.permute([2, 0, 1])
    image_tensor = transform(image_tensor)
    image_output1 = image_tensor.unsqueeze(0)

    image_tensor = torch.FloatTensor(inputs[1][0])
    image_tensor = image_tensor.permute([2, 0, 1])
    image_tensor = transform(image_tensor)
    image_output2 = image_tensor.unsqueeze(0)

    proprio_tensor = torch.FloatTensor(inputs[2][0].reshape(1, -1))
    proprio_output = proprio_tensor.unsqueeze(0)

    return [image_output1, image_output2, proprio_output]

def collect_done(step_obj, result):
    """Check agent done status"""
    for i, Id in enumerate(step_obj.agent_id):
        result[Id] = True
    return result

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
        self.fc1_1 = nn.Linear(8192, 128)
        # self.fc1_2 = nn.Linear(2048, 512)
        self.fc2_1 = nn.Linear(8192, 128)
        # self.fc2_2 = nn.Linear(2048, 512)
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(.3)
        self.dropout2 = nn.Dropout(.3)
        self.dropout3 = nn.Dropout(.3)
        self.dropout4 = nn.Dropout(.3)
        self.fc3 = nn.Linear(proprio_dim + 256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, action_dim)

    def forward(self, state):
        i_1 = no_pool_resnet(state[0]).view(state[0].size(0),-1)
        i_1 = self.dropout1(self.prelu(self.fc1_1(i_1)))
        # i_1 = self.tanh(self.fc1_2(i_1))
        i_2 = no_pool_resnet(state[1]).view(state[1].size(0),-1)
        i_2 = self.dropout2(self.prelu(self.fc2_1(i_2)))
        # i_2 = self.tanh(self.fc2_2(i_2))
        s = torch.cat([i_1, i_2, state[2].view(state[2].size(0), -1)], dim=1)
        s = self.dropout3(self.prelu(self.fc3(s)))
        s = self.dropout4(self.prelu(self.fc4(s)))
        return self.tanh(self.fc5(s))

class Critic(nn.Module):
    def __init__(self, proprio_dim):
        super(Critic, self).__init__()
        self.dropout1 = nn.Dropout(.3)
        self.dropout2 = nn.Dropout(.3)
        self.dropout3 = nn.Dropout(.3)
        self.dropout4 = nn.Dropout(.3)

        self.fc1_1 = nn.Linear(8192, 128)
        # self.fc1_2 = nn.Linear(2048, 512)
        self.fc2_1 = nn.Linear(8192, 128)
        # self.fc2_2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(proprio_dim + 256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 1)
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        i_1 = no_pool_resnet(state[0]).view(state[0].size(0),-1)
        i_1 = self.dropout1(self.prelu(self.fc1_1(i_1)))
        # i_1 = self.tanh(self.fc1_2(i_1))
        i_2 = no_pool_resnet(state[1]).view(state[1].size(0),-1)
        i_2 = self.dropout2(self.prelu(self.fc2_1(i_2)))
        # i_2 = self.tanh(self.fc2_2(i_2))
        s = torch.cat([i_1, i_2, state[2].view(state[2].size(0), -1)], dim=1)
        s = self.dropout3(self.prelu(self.fc3(s)))
        s = self.dropout4(self.prelu(self.fc4(s)))
        return self.fc5(s)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  Actor(state_dim, action_dim)
        # critic
        self.action_dim = action_dim
        self.critic = Critic(state_dim)
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def update_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std*new_action_std).to(device)

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

    def update_models_action_std(self, new_action_std):
        self.policy.update_action_std(new_action_std)
        self.policy_old.update_action_std(new_action_std)

    def update_models_eps_clip(self, new_eps_clip):
        self.eps_clip = new_eps_clip

    def select_action(self, state, memory):
        state = preprocess(state)
        state = [set.to(device) for set in state]
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
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = [torch.squeeze(torch.stack(state_set).to(device), 1).detach() for state_set in list(zip(*memory.states))]
        # old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
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
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards.float()) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
