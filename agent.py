import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as U
from collections import namedtuple, deque
from PPO_2_models import PPO_ActorCritic
import time
##### CONFIG PARMAS #####
##### CONFIG PARMAS #####
BATCH_SIZE = 512             # batch size of sampling
MIN_BATCH_NO = 32             # min no of batches needed in the memory before learning
GAMMA = 0.95                  # discount factor
T_MAX = 512                   # max number of time step for collecting trajectory
T_MAX_EPS = int(3e4)          # max number of steps before break
LR = 1e-4                     # learning rate #5e-4
OPTIM_EPSILON = 1e-5          # EPS for Adam optimizer
OPTIM_WGT_DECAY =  1e-4       # Weight Decay for Adam optimizer
GRAD_CLIP_MAX = 1.0           # max gradient allowed
CRITIC_L_WEIGHT = 1.0         # mean square error term weight
ENT_WEIGHT = 0.01             # weight of entropy added
ENT_DECAY = 0.99             # decay of entropy per 'step'
STD_SCALE_INIT = 2.0          # initial value of std scale for action resampling
STD_SCALE_DECAY = 0.99       # scale decay of std
P_RATIO_EPS = 0.1             # eps for ratio clip 1+eps, 1-eps
EPS_DECAY = 0.99           # decay factor for eps for ppo clip
NAN_PENALTY = -5.0            # penalty for actions that resulted in nan reward
USE_GAE = True                # use GAE flag
GAE_TAU = 0.99                # value control how much agent rely on current estimate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def saveTrainedModel(agent, path):
    state_dicts = {'actor_model': agent.model_local.actor.state_dict(),
                   'critic_model': agent.model_local.critic.state_dict()}
    torch.save(state_dicts, path)

def loadTrainedModel(agent, path):
    state_dicts = torch.load(path,map_location={'cuda:0': 'cpu'})

    agent.model_local.actor.load_state_dict(state_dicts['actor_model'])
    agent.model_local.critic.load_state_dict(state_dicts['critic_model'])

    return agent

def preprocess(inputs, agent):
    """Preprocess images and proprio input"""
    # image_tensor = torch.tensor(inputs.obs[0][agent])
    # image_tensor = image_tensor.permute([2, 0, 1])
    # image_batch1 = image_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    #
    # image_tensor = torch.tensor(inputs.obs[1][agent])
    # image_tensor = image_tensor.permute([2, 0, 1])
    # image_batch2 = image_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    proprio_tensor = torch.tensor(inputs.obs[0][agent])
    proprio_batch = proprio_tensor.unsqueeze(0)
    # print(proprio_batch)
    return [proprio_batch]

def collect_state(step_obj, result):
    for i, Id in enumerate(step_obj.agent_id):
        if not result[Id]:
            result[Id] = preprocess(step_obj, Id)
    return result

def collect_reward(step_obj, result):
    for i, Id in enumerate(step_obj.agent_id):
        if not result[Id]:
            result[Id] = step_obj.reward[i]
    return result

def collect_done(step_obj, result):
    for i, Id in enumerate(step_obj.agent_id):
        result[Id] = True
    return result

def any_false(lst):
    for i in lst:
        if isinstance(i, bool):
            return True
    return False

class PPO_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, env, image_size, proprio_size, action_size, num_agents=12, seed=0):
        """Initialize an Agent object.
        Params
        ======
            env (env object): object of the env
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents run in parallel
            seed (int): random seed
        """
        self.env = env
        self.brain_name = self.env.get_behavior_names()[0]
        self.proprio_size = proprio_size
        self.image_size = image_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.gamma = GAMMA # discount rate
        self.t_max = T_MAX # max number of steps for episodic exploration

        # Replay memory
        self.memory = ReplayBuffer(BATCH_SIZE, num_agents, seed)

        # Init Network Models and Optimizers
        self.model_local = PPO_ActorCritic(image_size, proprio_size, action_size, device, seed).to(device)
        self.optim = optim.Adam(self.model_local.parameters(), lr=LR,
                                eps=OPTIM_EPSILON, weight_decay=OPTIM_WGT_DECAY)

        # Initialize time step (for updating every UPDATE_EVERY steps and others)
        self.t_step = 1

        # entropy
        self.ent_weight = ENT_WEIGHT
        # eps for clipping
        self.p_ration_eps = P_RATIO_EPS
        # std for noise
        self.std_scale = STD_SCALE_INIT

        # for tracking
        self.total_steps = deque(maxlen=100)
        self.episodic_rewards = deque(maxlen=1000) # hist of rewards total of DONE episodes
        self.running_rewards = np.zeros(self.num_agents)

        self.critic_loss_hist = deque(maxlen=100)
        self.actor_gain_hist = deque(maxlen=100)

        # training or just accumulating experience?
        self.is_training = False

        print("current device: ", device)


    def _toTorch(self, s, dtype=torch.float32):
        return torch.tensor(s, dtype=dtype, device=device)

    def act(self, state):
        """Returns deterministic actions for given state using the
           Actor policy Network.
        Params
        ======
            state (array_like): current state, # agents x state_space
            action_values (array like, -1:+1) no grad
        """
        with torch.no_grad():
            _prob, _a_mean, action, _ent = self.model_local.actor(self._toTorch(state),
                                                                  std_scale=0.0)

        return np.clip(action.detach().cpu().numpy(), -1 , 1)


    def _collect_trajectory_data(self, train_mode=True, is_collecting=True):
        """
        Collect trajectory data and store them
        output: tuple of list (len: len(states)) of:
                states, log_probs, actions, rewards, As, rts
        """

        # reset env and running reward
        self.env.reset() # reset env
        self.running_rewards = np.zeros(self.num_agents)

        s, p, a, r, ns, d, A, V, rt = ([] for l in range(9)) #initialization
        states_recorded = [False] * self.num_agents
        self.env.step()
        step_result = self.env.get_steps(self.brain_name)
        states_recorded = collect_state(step_result[0], states_recorded)
        # while any_false(states_recorded):
        #     self.env.step()
        #     step_result = self.env.get_steps(self.brain_name)
        #     states_recorded = collect_state(step_result[0], states_recorded)

        state = [torch.cat(i, dim=0) for i in list(zip(*states_recorded))]
        # Collect the STEP trajectory data (s,a,r,ns,d)
        ep_len = 0
        while True:
            # state -> prob / actions

            state_predict = self.model_local(state, std_scale=self.std_scale)
            action = state_predict['a'] #torch, num_agents x action_size no grad
            # print(action)
            action = np.clip(action.detach().numpy(), -1., 1.)
            # if np.any(np.isnan(action.tolist())):
            #     print("nan action encountered!")
            #     return

            # env_info = self.env.step(action)[self.brain_name]
            self.env.set_actions(self.brain_name, action)
            self.env.step()

            done = [False] * self.num_agents
            states_recorded = [False] * self.num_agents
            rewards_recorded = [False] * self.num_agents

            step_result = self.env.get_steps(self.brain_name)
            states_recorded = collect_state(step_result[0], states_recorded)
            rewards_recorded = collect_reward(step_result[0], rewards_recorded)
            # done = collect_done(step_result[1], done)
             # array: (num_agents,)
            # while any_false(states_recorded):
            #     self.env.step()
            #     step_result = self.env.get_steps(self.brain_name)
            #     states_recorded = collect_state(step_result[0], states_recorded)
            #     rewards_recorded = collect_reward(step_result[0], rewards_recorded)
            #     done = collect_done(step_result[1], done)
            for i, v in enumerate(done):
                if v:
                    rewards_recorded[i] = -1

            reward = np.array(rewards_recorded)
            # print(states_recorded)
            next_state = [torch.cat(i, dim=0) for i in list(zip(*states_recorded))]
            #array: (num_agents,) boolean
            # recognize the current reward first
            # print(step_result[0].agent_id, step_result[1].agent_id, reward,  self.running_rewards)
            for i in step_result[0].agent_id:
                self.running_rewards[i] += reward[i]
            if is_collecting:
                s.append(state) #TENSOR: num_agents x state_size
                p.append(state_predict['log_prob'].detach()) #tensor: (num_agents x 1), NO grad
                a.append(state_predict['a']) #tensor: num_agents x action_size
                r.append(np.array(reward).reshape(-1,1)) #array: num_agents x 1
                ns.append(next_state) #TENSOR: num_agents x state_size
                d.append(1.*np.array(done).reshape(-1,1)) #array: num_agents x 1， 1. 0.
                V.append(state_predict['v'].detach().numpy()) #Q value TENSOR:
                                                              #num_agents x 1, require grad
            state = next_state
            if ep_len >= T_MAX:
                if is_collecting:
                    is_collecting = False
                    last_state = next_state

                # if np.all(done) or ep_len>=T_MAX_EPS: #only if t_max is reached and np.all done.
                    agents_mean_eps_reward = np.nanmean(self.running_rewards+1e-10)
                    if not np.isnan(agents_mean_eps_reward):
                        self.episodic_rewards.append(agents_mean_eps_reward) #avoid nan
                    self.total_steps.append(ep_len)
                    break

            ep_len += 1
        assert(len(s) == T_MAX+1)

        # Compute the Advantage/Return value
        # note that last state has no entry in record in V
        last_state_predict = self.model_local(last_state)

        # use td target as return, td error as advantage
        V.append(last_state_predict['v'])

        advantage = np.zeros([self.num_agents, 1])
        returns = last_state_predict['v'].detach().numpy()
        for i in reversed(range(T_MAX)):
            # effect of this loop is similar future credit assignments
            returns = r[i] + GAMMA * (1-d[i]) * returns
            #td_current = V[i].detach().numpy()
            if not USE_GAE:
                advantage = returns - V[i]
            else:
                td_error = r[i] + GAMMA * (1-d[i]) * V[i+1] - V[i]
                advantage = advantage * GAE_TAU * GAMMA * (1-d[i]) + td_error
            A.append(advantage) #array:, num_agents x 1, no grad
            rt.append(returns) #array:, num_agents x 1, no grad

        # reverse the list order
        A = A[::-1]
        rt = rt[::-1]

        # store data in memory
        self.memory.add((s, p, a, r, A, rt)) #tuple of list by types of data


    def step(self, train_mode=True):
        """ a step of collecting, sampling data and learn from it
            eps: for exploration if external noise is added
            train_mode: for the env
        """

        self._collect_trajectory_data(train_mode=train_mode)

        if train_mode and len(self.memory) >= BATCH_SIZE * MIN_BATCH_NO:
            if self.is_training == False:
                print("Prefetch completed. Training starts! \r")
                print("Number of Agents: ", self.num_agents)
                print("Device: ", device)
                self.is_training = True

            randomized_batches = self.memory.retrieve_memory() #sample from memory
            self.learn(randomized_batches) #learn from it and update grad

            # entropy weight decay
            self.ent_weight *= ENT_DECAY
            # std decay
            self.std_scale *= STD_SCALE_DECAY
            # eps decay
            self.p_ration_eps *= EPS_DECAY

            self.memory.reset()

        self.t_step += 1

    def learn(self, randomized_batches):
        """Update the parameters of the policy based on the data in the sampled
           trajectory.
        Params
        ======
        inputs:
            m_batch: (tuple) of:
                batch of states: (tensor) batch_size or num_agents x state_size
                batch of old_probs: (tensor) batch_size or num_agents x 1
                batch of actions: (tensor) batch_size or num_agents x state_size
                batch of rewards: (tensor) batch_size or num_agents x 1
                batch of Advantages: (tensor) batch_size or num_agents x 1
                batch of Returns/TDs: (tensor) batch_size or num_agents x 1
        """

        for (s, old_prob, old_actions, r, Advantage, returns_) in randomized_batches:
            #s, p, a, r, Advantage, td_target = m_batch

            ############################# ACTOR LOSS ##############################
            s_predictions = self.model_local(s, old_actions, self.std_scale) #use old s, a to get new prob
            new_prob = s_predictions['log_prob'] # num_agents x 1
            assert(new_prob.requires_grad == True)
            assert(Advantage.requires_grad == False)

            ratio = (new_prob - old_prob).exp() # # num_agent or m x 1

            G = ratio * Advantage

            G_clipped = torch.clamp(ratio, min=1.-P_RATIO_EPS,
                                           max=1.+P_RATIO_EPS) * Advantage

            G_loss = torch.min(G, G_clipped).mean(0)

            actor_loss = -(G_loss + self.ent_weight * s_predictions['ent'].mean())

            ############################ CRITIC LOSS ##############################
            assert(s_predictions['v'].requires_grad == True) #num_agent or m x 1, requires grad

            critic_loss = 0.5 * (returns_ - s_predictions['v']).pow(2).mean()

            # TOTAL LOSS
            total_loss = actor_loss + CRITIC_L_WEIGHT * critic_loss
            print("Total Loss:", total_loss, actor_loss, critic_loss)
            self.optim.zero_grad()
            total_loss.backward() #retain_graph=True
            U.clip_grad_norm_(self.model_local.parameters(), GRAD_CLIP_MAX)
            self.optim.step()

            self.actor_gain_hist.append(-actor_loss.data.detach().numpy())
            self.critic_loss_hist.append(critic_loss.data.detach().numpy())


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, batch_size, num_agents, seed=0):
        """Data Structure to store experience object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = []
        self.batch_size = batch_size
        self.num_agents = num_agents

        # data structure for storing individual experience
        self.data = namedtuple("data", field_names=["states", "old_probs",
                                                    "actions", "rewards",
                                                    "As", "returns"])
        torch.manual_seed(seed)


    def add(self, single_traj_data):
        """ Add a new experience to memory.
            data: (tuple) states, log_probs, rewards, As, Vs
        """
        (s_, p_, a_, r_, A_, rt_) = single_traj_data #equal lengths
        for s, p, a, r, A, rt in zip(s_, p_, a_, r_, A_, rt_): #by time step
            i = 0
            while i < self.num_agents: #by agent
                e = self.data([s[0][i]], p[i,:].detach(), a[i,:], r[i], A[i], rt[i])
                self.memory.append(e)
                i += 1

    def retrieve_memory(self):
        """Retrieve all data in memory in randomized order."""
        # convert memory structure into giant listS by data type
        (all_s, all_p, all_a, all_r, all_A, all_rt) = list(zip(*self.memory))
        assert(len(all_s) == len(self.memory))

        # so that we can normalized Advantage before sampling
        all_A = tuple((all_A - np.nanmean(all_A))/np.std(all_A))

        indices = np.arange(len(self.memory))
        np.random.shuffle(indices)
        indices = [indices[div*self.batch_size: (div+1)*self.batch_size]
                   for div in range(len(indices) // self.batch_size + 1)]

        result = []
        for batch_no, sample_ind in enumerate(indices):
            if len(sample_ind) >= self.batch_size / 2:
                s_s1, s_p, s_a, s_r, s_A, s_rt = ([] for l in range(6))

                i = 0
                while i < len(sample_ind): #while loop is faster
                    s_s1.append(all_s[sample_ind[i]][0]) #@each torch, state_size

                    # s_s2.append(all_s[sample_ind[i]][1]) #@each torch, state_size
                    # s_pr.append(all_s[sample_ind[i]][2]) #@each torch, state_size
                    s_p.append(all_p[sample_ind[i]]) #@each torch, 1
                    s_a.append(all_a[sample_ind[i]]) #@each array, (action_size,)
                    s_r.append(all_r[sample_ind[i]]) #@each array, 1
                    s_A.append(all_A[sample_ind[i]]) #@each array, 1
                    s_rt.append(all_rt[sample_ind[i]]) #@each array, 1
                    i += 1
                # change the format to tensor and make sure dims are correct for calculation
                s_s1 = torch.stack(s_s1).float().squeeze(1)
                # s_s2 = torch.stack(s_s2).float().squeeze(1)
                # s_pr = torch.stack(s_pr).float().squeeze(1)
                s_p = torch.stack(s_p).float()
                s_a = torch.stack(s_a).float()
                s_r = torch.from_numpy(np.vstack(s_r)).float()
                s_A = torch.from_numpy(np.stack(s_A)).float()
                s_rt = torch.from_numpy(np.stack(s_rt)).float()

                result.append(([s_s1], s_p, s_a, s_r, s_A, s_rt))

        return result


    def reset(self):
        self.memory = []

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
