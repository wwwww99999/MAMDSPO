import time
from statistics import mean
from tqdm import tqdm
import safety_gymnasium
import torch
import numpy as np
from net import GaussianPolicy, Value
from agent import Agent
from data_generator import DataGenerator
import csv
from normalize import Normalize
import re

def train(env_name, batch_size, gae_r_lambda, gae_c_lambda, r_gamma, c_gamma, max_iter_num,
          epoch, minibatch, policy_lr, r_value_lr, c_value_lr, l2_reg, hidden_dim,
          eta_kl, target_kl, lam, lam_max, lam_lr, cost_threshold, seed, activation, device):

    if 'Velocity' in env_name:
        pattern = r'Safety(\d+x\d+)([A-Za-z]+)Velocity-v0'
        match = re.match(pattern, env_name)
        env = safety_gymnasium.make_ma(match.group(2), match.group(1))
    else:
        env = safety_gymnasium.make(env_name)
    agents = env.agents
    max_episode_steps = env._max_episode_steps
    agent_dims = {}

    # 计算全局状态维度（所有智能体观测拼接）
    global_state_dim = 0
    for agent_name in agents:
        state_dim = env.observation_space(agent_name).shape[0]
        action_dim = env.action_space(agent_name).shape[0]
        agent_dims[agent_name] = {'state_dim': state_dim, 'action_dim': action_dim}
        global_state_dim += state_dim

    # 创建每个智能体的策略和价值网络
    policies = {}
    r_values = {}
    c_values = {}

    for agent_name in agents:
        dims = agent_dims[agent_name]

        policies[agent_name] = GaussianPolicy(dims['state_dim'], dims['action_dim'],
            hidden_sizes=(hidden_dim, hidden_dim), activation=activation).to(device)

        r_values[agent_name] = Value(global_state_dim, hidden_sizes=(hidden_dim, hidden_dim),
            activation=activation).to(device)

        c_values[agent_name] = Value(global_state_dim, hidden_sizes=(hidden_dim, hidden_dim),
            activation=activation).to(device)

    env.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    normalize = {agent_name: Normalize(clip=5) for agent_name in agents}

    agents_dict = {}
    for agent_name in agents:
        agents_dict[agent_name] = Agent(
            policies[agent_name], r_values[agent_name], c_values[agent_name],
            epoch, minibatch, policy_lr, r_value_lr, c_value_lr,
            l2_reg, eta_kl, target_kl, c_gamma, lam, lam_max, lam_lr,
            cost_threshold, device
        )

    data_generator = DataGenerator(agents, agent_dims, global_state_dim, batch_size,
                                   max_episode_steps, env, env_name, gae_r_lambda,
                                   gae_c_lambda, r_gamma, c_gamma, normalize, seed, device)

    agent_rewards = {agent_name: [] for agent_name in agents}
    agent_costs = {agent_name: [] for agent_name in agents}

    start = time.time()

    for j in range(10):
        with tqdm(total=int(max_iter_num / 10), desc='Iteration %d' % (j + 1)) as pbar:
            for i_episode in range(int(max_iter_num / 10)):
                interact_data = data_generator.interact(policies, r_values, c_values)
                for agent_name in agents:
                    agent_rewards[agent_name].append(interact_data[agent_name]['average_episode_r'])
                    agent_costs[agent_name].append(interact_data[agent_name]['average_episode_c'])
                    agents_dict[agent_name].update(interact_data[agent_name], interact_data['global_states'])

                if (i_episode + 1) % 10 == 0:
                    postfix_info = {
                        'episode': '%d' % (max_iter_num / 10 * j + i_episode + 1)
                    }

                    for agent_name in agents:
                        r_list = agent_rewards[agent_name]
                        c_list = agent_costs[agent_name]
                        avg_r = mean(r_list[-10:])
                        avg_c = mean(c_list[-10:])
                        postfix_info[f'{agent_name}_r'] = '%.3f' % avg_r
                        postfix_info[f'{agent_name}_c'] = '%.3f' % avg_c

                    all_avg_r = mean([agent_rewards[a][-1] for a in agents])
                    all_avg_c = mean([agent_costs[a][-1] for a in agents])
                    postfix_info['avg_r'] = '%.3f' % all_avg_r
                    postfix_info['avg_c'] = '%.3f' % all_avg_c

                    pbar.set_postfix(postfix_info)
                pbar.update(1)

    end = time.time()
    print("Training Time:" '%.2f' %(end - start))

    # 保存每个智能体的结果到CSV
    for agent_name in agents:
        with open(f"r_{agent_name}_{seed}.csv", "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(agent_rewards[agent_name])

        with open(f"c_{agent_name}_{seed}.csv", "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(agent_costs[agent_name])


    for agent_name in agents:
        torch.save({
            'policy': policies[agent_name].state_dict(),
            'r_value': r_values[agent_name].state_dict(),
            'c_value': c_values[agent_name].state_dict(),
        }, f'agent_{agent_name}_{seed}.pth')
