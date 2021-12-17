#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-10-07 20:57:11
LastEditor: John
LastEditTime: 2021-09-23 12:23:01
Discription: 
Environment: 
'''
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import numpy as np


def chinese_font():
    return FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)  # 系统字体路径，此处是mac的


def plot_rewards(rewards, ma_rewards, tag="train", env='CartPole-v0', algo="DQN", save=True, path='./'):
    sns.set()
    plt.title("average learning curve of {} for {}".format(algo, env))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if save:
        plt.savefig(path + "{}_rewards_curve".format(tag))
    plt.show()


def plot_rewards_cn(rewards, ma_rewards, tag="train", env='CartPole-v0', algo="DQN", save=True, path='./'):
    ''' 中文画图
    '''
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(env, algo), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_rewards_curve_cn")
    # plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()


def plot_speed(total_v_list, total_t_list, total_a_list, tag="train", env='Train Optimal', algo="DDPG", save=True,
               path='./'):
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的速度曲线".format(env, algo), fontproperties=chinese_font())
    ax1 = plt.axes(projection='3d')
    for i in range(len(total_v_list)):
        if i % 20 == 0:
            a = np.array(total_v_list[i]).reshape(-1)
            b = np.array(total_t_list[i]).reshape(-1)
            c = np.linspace(1, len(total_t_list[i]), len(total_t_list[i]))
            ax1.plot3D(b, c, a)
    plt.legend((u'速度曲线',), loc="best", prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_speed_profile_cn")
    plt.figure()
    for i in range(len(total_a_list)):
        if i % 20 == 0:
            plt.plot(total_a_list[i])
    plt.legend((u'动作曲线',), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_action_cn")
    plt.show()


def evalplot_speed(total_v_list, total_t_list, total_a_list, tag="eval", env='Train Optimal', algo="DDPG", save=True,
                   path='./'):
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的速度曲线".format(env, algo), fontproperties=chinese_font())
    ax1 = plt.axes(projection='3d')
    a = np.array(total_v_list[0]).reshape(-1)
    b = np.array(total_t_list[0]).reshape(-1)
    c = np.linspace(1, len(total_t_list[0]), len(total_t_list[0]))
    ax1.plot3D(b, c, a)
    plt.legend((u'速度曲线',), loc="best", prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_speed_profile_cn")
    plt.figure()
    plt.plot(total_a_list[0])
    plt.legend((u'动作曲线',), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_action_cn")
    plt.show()
