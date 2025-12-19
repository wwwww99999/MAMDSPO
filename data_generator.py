from statistics import mean
import numpy as np
import torch
from environment import get_velocity_threshold

class DataGenerator:
    def __init__(self, agents, agent_dims, global_state_dim, batch_size, max_episode_steps, env,
                 env_name, gae_r_lambda, gae_c_lambda, r_gamma, c_gamma, normalize, seed, device):
        self.agents = agents
        self.agent_dims = agent_dims
        self.global_state_dim = global_state_dim
        self.batch_size = batch_size
        self.max_episode_steps = max_episode_steps
        self.env = env
        self.env_name = env_name
        self.gae_r_lambda = gae_r_lambda
        self.gae_c_lambda = gae_c_lambda
        self.r_gamma = r_gamma
        self.c_gamma = c_gamma
        self.normalize = normalize
        self.seed = seed
        self.device = device

        # 为每个智能体初始化数据结构
        self.agent_data = {}
        for agent_name in agents:
            state_dim = agent_dims[agent_name]['state_dim']
            action_dim = agent_dims[agent_name]['action_dim']

            self.agent_data[agent_name] = {
                # Batch buffers
                'state_buf': np.zeros((batch_size, state_dim), dtype=np.float32),
                'action_buf': np.zeros((batch_size, action_dim), dtype=np.float32),
                'adv_r_buf': np.zeros((batch_size, 1), dtype=np.float32),
                'adv_c_buf': np.zeros((batch_size, 1), dtype=np.float32),
                'td_r_target_buf': np.zeros((batch_size, 1), dtype=np.float32),
                'td_c_target_buf': np.zeros((batch_size, 1), dtype=np.float32),

                # Episode buffers
                'state_eps': np.zeros((max_episode_steps, state_dim), dtype=np.float32),
                'next_state_eps': np.zeros((max_episode_steps, state_dim), dtype=np.float32),
                'action_eps': np.zeros((max_episode_steps, action_dim), dtype=np.float32),
                'reward_eps': np.zeros((max_episode_steps, 1), dtype=np.float32),
                'cost_eps': np.zeros((max_episode_steps, 1), dtype=np.float32),
                'done_eps': np.zeros((max_episode_steps, 1), dtype=np.bool8),
            }

        # 全局状态buffer（所有智能体共享）
        self.global_state_buf = np.zeros((batch_size, global_state_dim), dtype=np.float32)
        self.global_next_state_buf = np.zeros((batch_size, global_state_dim), dtype=np.float32)

        self.episode_steps = 0

    def interact(self, policies, r_values, c_values):
        batch_idx = 0
        agent_average_r = {agent_name: [] for agent_name in self.agents}
        agent_average_c = {agent_name: [] for agent_name in self.agents}
        while batch_idx < self.batch_size:
            state_dict = self.env.reset()[0]
            for agent_name in self.agents:
                state_dict[agent_name] = self.normalize[agent_name].normalize(state_dict[agent_name])

            agent_episode_r = {agent_name: 0.0 for agent_name in self.agents}
            agent_episode_c = {agent_name: 0.0 for agent_name in self.agents}
            for t in range(self.max_episode_steps):
                action_dict = {agent_name: policies[agent_name].take_action(torch.tensor(state_dict[agent_name], dtype=torch.float)
                                                                            .to(self.device)) for agent_name in self.agents}
                next_state_dict, r_dict, c_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)
                if 'Velocity' in self.env_name:
                    v_threshold = get_velocity_threshold(self.env_name)
                    for agent_name in self.agents:
                        if 'y_velocity' not in info_dict[agent_name]:
                            velocity = np.abs(info_dict[agent_name]['x_velocity'])
                        else:
                            velocity = np.sqrt(info_dict[agent_name]['x_velocity'] ** 2 + info_dict[agent_name]['y_velocity'] ** 2)
                        if velocity > v_threshold:
                            c_dict[agent_name] = 1
                        else:
                            c_dict[agent_name] = 0

                for agent_name in self.agents:
                    agent_episode_r[agent_name] += r_dict[agent_name]
                    agent_episode_c[agent_name] += c_dict[agent_name]
                    next_state_dict[agent_name] = self.normalize[agent_name].normalize(next_state_dict[agent_name])

                self.global_state_buf[batch_idx] = np.concatenate([state_dict[agent_name] for agent_name in self.agents])
                self.global_next_state_buf[batch_idx] = np.concatenate([next_state_dict[agent_name] for agent_name in self.agents])

                # Store in episode buffer
                for agent_name in self.agents:
                    self.agent_data[agent_name]['state_eps'][t] = state_dict[agent_name]
                    self.agent_data[agent_name]['next_state_eps'][t] = next_state_dict[agent_name]
                    self.agent_data[agent_name]['action_eps'][t] = action_dict[agent_name]
                    self.agent_data[agent_name]['reward_eps'][t] = r_dict[agent_name]
                    self.agent_data[agent_name]['cost_eps'][t] = c_dict[agent_name]
                    self.agent_data[agent_name]['done_eps'][t] = terminated_dict[agent_name]

                state_dict = next_state_dict
                batch_idx += 1
                self.episode_steps += 1

                if any(terminated_dict.values()) or any(truncated_dict.values()):
                    for agent_name in self.agents:
                        agent_average_r[agent_name].append(agent_episode_r[agent_name])
                        agent_average_c[agent_name].append(agent_episode_c[agent_name])
                    break

                if batch_idx == self.batch_size:
                    break

            start_idx = batch_idx - self.episode_steps
            end_idx = batch_idx

            for agent_name in self.agents:
                self.agent_data[agent_name]['state_eps'] = self.agent_data[agent_name]['state_eps'][:self.episode_steps]
                self.agent_data[agent_name]['next_state_eps'] = self.agent_data[agent_name]['next_state_eps'][:self.episode_steps]
                self.agent_data[agent_name]['action_eps'] = self.agent_data[agent_name]['action_eps'][:self.episode_steps]
                self.agent_data[agent_name]['reward_eps'] = self.agent_data[agent_name]['reward_eps'][:self.episode_steps]
                self.agent_data[agent_name]['cost_eps'] = self.agent_data[agent_name]['cost_eps'][:self.episode_steps]
                self.agent_data[agent_name]['done_eps'] = self.agent_data[agent_name]['done_eps'][:self.episode_steps]

                adv_r, adv_c, td_r_target, td_c_target = self.GAE(agent_name, start_idx, end_idx, r_values[agent_name], c_values[agent_name])

                # Update batch buffer
                self.agent_data[agent_name]['state_buf'][start_idx: end_idx] = self.agent_data[agent_name]['state_eps']
                self.agent_data[agent_name]['action_buf'][start_idx: end_idx] = self.agent_data[agent_name]['action_eps']
                self.agent_data[agent_name]['adv_r_buf'][start_idx: end_idx] = adv_r
                self.agent_data[agent_name]['adv_c_buf'][start_idx: end_idx] = adv_c
                self.agent_data[agent_name]['td_r_target_buf'][start_idx: end_idx] = td_r_target
                self.agent_data[agent_name]['td_c_target_buf'][start_idx: end_idx] = td_c_target

                # Reset episode buffer and update pointer
                state_dim = self.agent_dims[agent_name]['state_dim']
                action_dim = self.agent_dims[agent_name]['action_dim']
                self.agent_data[agent_name]['state_eps'] = np.zeros((self.max_episode_steps, state_dim), dtype=np.float32)
                self.agent_data[agent_name]['next_state_eps'] = np.zeros((self.max_episode_steps, state_dim), dtype=np.float32)
                self.agent_data[agent_name]['action_eps'] = np.zeros((self.max_episode_steps, action_dim), dtype=np.float32)
                self.agent_data[agent_name]['reward_eps'] = np.zeros((self.max_episode_steps, 1), dtype=np.float32)
                self.agent_data[agent_name]['cost_eps'] = np.zeros((self.max_episode_steps, 1), dtype=np.float32)
                self.agent_data[agent_name]['done_eps'] = np.zeros((self.max_episode_steps, 1), dtype=np.bool8)

            self.episode_steps = 0

        # Normalize advantage functions
        for agent_name in self.agents:
            adv_r_buf = self.agent_data[agent_name]['adv_r_buf']
            self.agent_data[agent_name]['adv_r_buf'] = (adv_r_buf - adv_r_buf.mean()) / (adv_r_buf.std() + 1e-6)
            adv_c_buf = self.agent_data[agent_name]['adv_c_buf']
            self.agent_data[agent_name]['adv_c_buf'] = (adv_c_buf - adv_c_buf.mean()) / (adv_c_buf.std() + 1e-6)

        result = {}

        for agent_name in self.agents:
            result[agent_name] = {
                'states': self.agent_data[agent_name]['state_buf'],
                'actions': self.agent_data[agent_name]['action_buf'],
                'r_advantages': self.agent_data[agent_name]['adv_r_buf'],
                'c_advantages': self.agent_data[agent_name]['adv_c_buf'],
                'td_r_targets': self.agent_data[agent_name]['td_r_target_buf'],
                'td_c_targets': self.agent_data[agent_name]['td_c_target_buf'],
                'average_episode_r': mean(agent_average_r[agent_name]),
                'average_episode_c': mean(agent_average_c[agent_name]),
            }

        result['global_states'] = self.global_state_buf

        return result

    def GAE(self, agent_name, start_idx, end_idx, r_value, c_value):
        adv_r = np.zeros((self.episode_steps, 1))
        prev_adv_r = 0
        adv_c = np.zeros((self.episode_steps, 1))
        prev_adv_c = 0
        v_r_t = r_value(torch.tensor(self.global_state_buf[start_idx: end_idx], dtype=torch.float).to(self.device)).detach().cpu().numpy()
        v_r_tplus1 = r_value(torch.tensor(self.global_next_state_buf[start_idx: end_idx], dtype=torch.float).to(self.device)).detach().cpu().numpy()
        v_c_t = c_value(torch.tensor(self.global_state_buf[start_idx: end_idx], dtype=torch.float).to(self.device)).detach().cpu().numpy()
        v_c_tplus1 = c_value(torch.tensor(self.global_next_state_buf[start_idx: end_idx], dtype=torch.float).to(self.device)).detach().cpu().numpy()
        td_r_delta = self.agent_data[agent_name]['reward_eps'] + self.r_gamma * v_r_tplus1 * (1 - self.agent_data[agent_name]['done_eps']) - v_r_t
        td_c_delta = self.agent_data[agent_name]['cost_eps'] + self.c_gamma * v_c_tplus1 * (1 - self.agent_data[agent_name]['done_eps']) - v_c_t
        for t in reversed(range(self.episode_steps)):
            adv_r[t] = td_r_delta[t] + self.r_gamma * self.gae_r_lambda * prev_adv_r
            adv_c[t] = td_c_delta[t] + self.c_gamma * self.gae_c_lambda * prev_adv_c
            prev_adv_r = adv_r[t]
            prev_adv_c = adv_c[t]
        td_r_target = v_r_t + adv_r
        td_c_target = v_c_t + adv_c
        return adv_r, adv_c, td_r_target, td_c_target