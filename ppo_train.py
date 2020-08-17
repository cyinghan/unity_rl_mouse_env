import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ppo_actor_critic import *
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

def main():
    ############## Hyperparameters ##############
    env_name = "mouse_agent-v2"
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 10           # print avg reward in the interval
    max_episodes = 3000        # max training episodes
    max_timesteps = 600        # max timesteps in one episode

    update_timestep = 1200      # update policy every n timesteps
    action_std = 0.25           # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr = 0.0001                 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    # creating environment
    env = UnityEnvironment(base_port=5004)
    env.reset()
    group_name = env.get_behavior_names()[0]
    group_spec = env.get_behavior_spec(group_name)
    step_result = env.get_steps(group_name)
    print("Number of observations : ", group_spec.observation_shapes)
    action_dim = 3
    state_dim = 62

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy.load_state_dict(torch.load('./PPO_continuous_mouse_agent-v2.pth'))
    print(lr,betas)

    # logging variables
    best_reward = 0
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = step_result[0].obs
        episode_reward = 0
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            env.set_actions(group_name, np.array([np.clip(action, -1, 1)]))
            env.step()
            step_result = env.get_steps(group_name)
            done = collect_done(step_result[1], [False])[0]
            if done:
                print("Episode terminated early...")
                reward = 0
            else:
                state = step_result[0].obs
                reward = step_result[0].reward[0]

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)


            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            episode_reward += reward
            # if render:
            #     env.render()
            if done:
                break
        avg_length += t
        writer.add_scalar('Average Reward', episode_reward, i_episode)
        writer.flush()
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            # torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:

            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            if running_reward > best_reward:
                best_reward = running_reward
                torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
        env.reset()
        env.step()
        step_result = env.get_steps(group_name)

if __name__ == '__main__':
    writer = SummaryWriter()
    main()
    writer.close()
