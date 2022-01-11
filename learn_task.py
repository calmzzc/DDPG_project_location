#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2021-09-16 01:31:33
@Discription:
@Environment: python 3.7.7
'''
import sys, os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加父路径到系统路径sys.path

import datetime
import gym
import torch

from env import NormalizedActions, OUNoise
from agent import DDPG
from utils import save_results, make_dir
from plot import plot_rewards, plot_rewards_cn, plot_speed, evalplot_speed, plot_trainep_speed, plot_evalep_speed, \
    plot_power_cn, plot_unsafecounts_cn
from environment import Line

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'  # 算法名称
        self.env = 'Optimal_Control'  # 环境名称
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.train_eps = 500  # 测试的回合数
        self.eval_eps = 30  # 测试的回合数
        self.gamma = 0.99  # 折扣因子
        self.critic_lr = 1e-3  # 评论家网络的学习率
        self.actor_lr = 1e-4  # 演员网络的学习率
        self.memory_capacity = 8000
        self.batch_size = 32
        self.target_update = 2
        self.hidden_dim = 256
        self.soft_tau = 1e-2  # 软更新参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_agent_config(cfg, seed=1):
    # env = NormalizedActions(gym.make(cfg.env))
    env = Line()
    # env.seed(seed)  # 随机种子
    state_dim = 2
    action_dim = 1
    agent = DDPG(state_dim, action_dim, cfg)
    return env, agent


def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env}，算法：{cfg.algo}，设备：{cfg.device}')
    ou_noise = OUNoise(env.action_space)  # 动作噪声
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    unsafe_counts = []  # 记录超速次数
    ma_unsafe_counts = []  # 记录滑动平均次数
    total_t_list = []
    total_x_list = []
    total_v_list = []
    total_a_list = []
    total_oa_list = []
    total_ep_list = []
    total_power_list = []
    ma_total_power_list = []
    for i_ep in range(cfg.train_eps):
        total_ep_list.append(i_ep)
        state = env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        ep_unsafe_counts = 0
        i_step = 0
        x_list = []
        t_list = [0]
        v_list = [0]
        a_list = [0]
        oa_list = [0]
        total_power = 0
        while True:
            i_step += 1
            action = agent.choose_action(state)
            oa_list.append(action)
            action = ou_noise.get_action(action, i_step)

            next_state, reward, done, time, velocity, total_power, action, ep_unsafe_counts = env.step(total_power,
                                                                                                       state,
                                                                                                       action, i_step,
                                                                                                       ep_unsafe_counts)

            t_list.append(time)
            v_list.append(velocity)
            a_list.append(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if done:
                total_t_list.append(t_list.copy())
                total_v_list.append(v_list.copy())
                total_a_list.append(a_list.copy())
                total_oa_list.append(oa_list.copy())
                t_list.clear()
                v_list.clear()
                a_list.clear()
                oa_list.clear()
                break
        if (i_ep + 1) % 10 == 0:
            print('回合：{}/{}，奖励：{}, 能耗  {}, 最终时间  {}, 最终速度  {}, 最终位置  {},不安全次数  {}'.format(i_ep + 1,
                                                                                          cfg.train_eps,
                                                                                          np.around(ep_reward[0], 2),
                                                                                          np.around(total_power[0], 2),
                                                                                          np.around(time[0], 2), np.
                                                                                          around(velocity[0], 2),
                                                                                          i_step * 50,
                                                                                          np.round(ep_unsafe_counts,
                                                                                                   0)))
        rewards.append(ep_reward)
        unsafe_counts.append(ep_unsafe_counts)
        if ma_unsafe_counts:
            ma_unsafe_counts.append(0.9 * ma_unsafe_counts[-1] + 0.1 * ep_unsafe_counts)
        else:
            ma_unsafe_counts.append(ep_unsafe_counts)
        total_power_list.append(total_power)
        if ma_total_power_list:
            ma_total_power_list.append(0.9 * ma_total_power_list[-1] + 0.1 * total_power)
        else:
            ma_total_power_list.append(total_power)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练！')
    return rewards, ma_rewards, total_v_list, total_t_list, total_a_list, total_ep_list, total_power_list, ma_total_power_list, unsafe_counts, ma_unsafe_counts


def train2(cfg, env, agent):
    print('开始训练!')
    print(f'环境：{cfg.env}, 算法：{cfg.algo}, 设备：{cfg.device}')
    ou_noise = OUNoise(env.action_space)  # 动作噪声
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    total_t_list = []
    total_v_list = []
    total_a_list = []
    i_ep = 0

    while i_ep < cfg.train_eps:
        i_step = 0
        x_list = []
        t_list = [0]
        v_list = [0]
        a_list = [0]
        oa_list = [0]
        state = env.reset()
        done = False
        ep_reward = 0
        total_power = 0
        tc_location = 0
        while True:
            i_step += 1
            action = agent.choose_action(state)
            oa_list.append(action)
            action = ou_noise.get_action(action, i_step)
            next_state, reward, done, time, velocity, total_power, action = env.step2(total_power, state,
                                                                                      action, i_step)
            if done == 1:
                c_index = done
                t_list.append(time)
                v_list.append(velocity)
                a_list.append(action)
                ep_reward += reward
                agent.memory.push(state, action, reward, next_state, done)
                agent.update()
                total_t_list.append(t_list.copy())
                total_v_list.append(v_list.copy())
                total_a_list.append(a_list.copy())
                t_list.clear()
                v_list.clear()
                a_list.clear()
                i_ep = i_ep + 1
                break
            elif done == 2:
                c_index = done
                t_list.clear()
                v_list.clear()
                a_list.clear()
                break
            elif done == 0:
                t_list.append(time)
                v_list.append(velocity)
                a_list.append(action)
                ep_reward += reward
                agent.memory.push(state, action, reward, next_state, done)
                state = next_state
                agent.update()
        if i_ep >= 0:
            if c_index == 2:
                #  print('当前回合无效')
                continue
            if (i_ep + 1) % 10 == 0:
                print('回合：{}/{}, 奖励：{}, 能耗  {}, 最终时间  {}, 最终速度  {}, 最终位置  {}'.format(i_ep + 1, cfg.train_eps,
                                                                                     np.around(ep_reward[0], 2),
                                                                                     np.around(total_power[0], 2),
                                                                                     np.around(time[0], 2),
                                                                                     np.around(velocity[0], 2),
                                                                                     np.around(i_step * 50, 2)))
        rewards.append(ep_reward)
        # save ma_rewards
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练！')
    return rewards, ma_rewards, total_v_list, total_t_list, total_a_list


def eval(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env}, 算法：{cfg.algo}, 设备：{cfg.device}')
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    unsafe_counts = []  # 记录超速次数
    ma_unsafe_counts = []  # 记录滑动平均次数
    total_x_list = []
    total_t_list = []
    total_v_list = []
    total_a_list = []
    total_ep_list = []
    for i_ep in range(cfg.eval_eps):
        total_ep_list.append(i_ep)
        state = env.reset()
        # state = env.eval_reset()
        done = False
        ep_unsafe_counts = 0
        ep_reward = 0
        i_step = 0
        total_power = 0
        x_list = []
        v_list = [0]
        t_list = [0]
        a_list = [0]
        while True:
            i_step += 1
            action = agent.choose_action(state)
            action = np.array(action).reshape(1)
            next_state, reward, done, time, velocity, total_power, action, ep_unsafe_counts = env.step(total_power,
                                                                                                       state,
                                                                                                       action, i_step,
                                                                                                       ep_unsafe_counts)
            t_list.append(time)
            v_list.append(velocity)
            a_list.append(action)
            ep_reward += reward
            state = next_state
            if done:
                total_t_list.append(t_list.copy())
                total_v_list.append(v_list.copy())
                total_a_list.append(a_list.copy())
                t_list.clear()
                v_list.clear()
                a_list.clear()
                break
        print('回合：{}/{}，奖励：{}, 能耗  {}, 最终时间  {}, 最终速度  {}, 最终位置  {}'.format(i_ep + 1,
                                                                            cfg.eval_eps,
                                                                            np.around(ep_reward[0], 2),
                                                                            np.around(total_power[0], 2),
                                                                            np.around(time[0], 2), np.
                                                                            around(velocity[0], 2),
                                                                            i_step))
        rewards.append(ep_reward)
        unsafe_counts.append(ep_unsafe_counts)
        if ma_unsafe_counts:
            ma_unsafe_counts.append(0.9 * unsafe_counts[-1] + 0.1 * ep_unsafe_counts)
        else:
            ma_unsafe_counts.append(ep_unsafe_counts)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成测试！')
    return rewards, ma_rewards, total_v_list, total_t_list, total_a_list, total_ep_list


if __name__ == "__main__":
    cfg = DDPGConfig()
    # 训练
    env, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards, v_list, t_list, a_list, ep_list, power_list, ma_power_list, unsafe_c, ma_unsafe_c = train(cfg,
                                                                                                                   env,
                                                                                                                   agent)
    # rewards, ma_rewards, v_list, t_list, a_list = train2(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
    plot_power_cn(power_list, ma_power_list, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
    # 测试
    env, agent = env_agent_config(cfg, seed=1)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards, ev_list, et_list, ea_list, eval_ep_list = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)

    plot_speed(v_list, t_list, a_list, tag="op_train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
    evalplot_speed(ev_list, et_list, ea_list, tag="op_eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)

    plot_trainep_speed(v_list, t_list, a_list, ep_list, tag="ep_train", env=cfg.env, algo=cfg.algo,
                       path=cfg.result_path)
    plot_evalep_speed(ev_list, et_list, ea_list, eval_ep_list, tag="ep_eval", env=cfg.env, algo=cfg.algo,
                      path=cfg.result_path)
    plot_unsafecounts_cn(unsafe_c, ma_unsafe_c, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
