import os
import matplotlib.pyplot as plt
import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from agent import *

channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name='test_run', side_channels=[channel])
channel.set_configuration_parameters(time_scale = 3.0)

#Reset the environment
env.reset()
# Set the default brain to work with
group_name = env.get_behavior_names()[0]
group_spec = env.get_behavior_spec(group_name)
step_result = env.get_steps(group_name)

print("Number of observations : ", group_spec.observation_shapes)
num_agents = 1
agent = PPO_Agent(env, 128, 3, group_spec.action_size, num_agents)

model_dir = 'saved_models/'
model_name = 'unity_continuous_' + str(group_name) + '_' + str(num_agents) + '_maxscore' + '_agents.pt'

episode_max = 2000 # training loop max iterations
episode_reward = 0.0
mean_rewards = []
max_score = -np.inf
e = 0
goal_reached = False

while e < episode_max:

    # collect trajectories
    agent.step()
    episode_reward = agent.episodic_rewards

    # display some progress every 20 iterations
    if agent.is_training:

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(episode_reward))

        if (e+5)%1==0 :
            print("e: {}  score: {:.2f}  Avg score(100e): {:.2f}  "
                  "A(g): {:.2f}  C(l): {:.2f}  std: {:.2f}  steps: {}".format(e+1, np.mean(episode_reward),
                                                                              np.mean(mean_rewards[-100:]),
                                                                              np.mean(agent.actor_gain_hist),
                                                                              np.mean(agent.critic_loss_hist),
                                                                              agent.std_scale,
                                                                              int(np.mean(agent.total_steps))))
        if np.mean(mean_rewards[-100:]) > max_score:
            max_score = np.mean(mean_rewards[-100:])
            saveTrainedModel(agent, model_dir + model_name)

        if np.mean(episode_reward) >= 2000 and not goal_reached :
            goal_reached = True
            episode_max = e + 5
            print("Score reached benchmark of 2000. Problem Solved!")

        e += 1
    else:
        print('\rFetching experiences... {} '.format(len(agent.memory.memory)), end="")
