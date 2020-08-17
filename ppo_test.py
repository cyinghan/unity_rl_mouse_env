
from PIL import Image
import torch
import numpy as np
from ppo_actor_critic import *

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    ############## Hyperparameters ##############
    env_name = "mouse_agent-v2"
    env = UnityEnvironment(base_port=5004)
    env.reset()
    group_name = env.get_behavior_names()[0]
    group_spec = env.get_behavior_spec(group_name)
    step_result = env.get_steps(group_name)
    state_dim = 62
    action_dim = 3

    n_episodes = 5          # num of episodes to run
    max_timesteps = 600    # max timesteps in one episode

    # filename and directory to load model from
    filename = 'PPO_continuous_mouse_agent-v2.pth'
    directory = "./"

    action_std = 0.25        # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory+filename))

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = step_result[0].obs
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            env.set_actions(group_name, np.array([np.clip(action, -1, 1)]))
            env.step()
            step_result = env.get_steps(group_name)
            done = collect_done(step_result[1], [False])[0]
            reward =  step_result[0].reward[0]
            state = step_result[0].obs
            ep_reward += reward
            if done:
                break

        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.reset()
        env.step()
        step_result = env.get_steps(group_name)

if __name__ == '__main__':
    test()
