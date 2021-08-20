import sys, os, time
import numpy as np
import math
import gym
from ruamel.yaml import YAML
import envs

import torch
from common.sac import SAC
from common.sarc import SARC
from baselines.discrim import ResNetAIRLDisc, MLPDisc
from baselines.adv_smm import AdvSMM

from utils import system, collect, logger
import datetime
import dateutil.tz
import json

def evaluate_policy(policy, env, n_episodes, deterministic=False):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    expert_states = torch.zeros((n_episodes, env_T, obs_dim)) # s1 to sT
    expert_actions = torch.zeros((n_episodes, env_T, act_dim)) # a0 to aT-1

    returns = []

    # t_max = 0 # max episode seen at evaluation
    for n in range(n_episodes):
        print(n)
        obs = env.reset()
        ret = 0
        t = 0
        done = False
        stored_terminal_rew = False
        # while not done and t<env_T:
        while t<env_T:
            # print(t)
            action = policy(obs, deterministic)
            # action = np.array([1,0])
            obs, rew, done, _ = env.step(action) # NOTE: assume rew=0 after done=True for evaluation
            if done:
                rew = 0 if stored_terminal_rew else rew
                stored_terminal_rew = True
            expert_states[n, t, :] = torch.from_numpy(obs).clone()
            expert_actions[n, t, :] = torch.from_numpy(action).clone()
            ret += rew
            t += 1
            env.render()
            # time.sleep(0.1)
            
            # t_max = max(t_max,t)
        returns.append(ret)

    # return expert_states[:,:t_max], expert_actions[:,:t_max], np.array(returns) # clip to max episode length seen at evaluation
    return expert_states, expert_actions, np.array(returns)

if __name__ == "__main__":
    yaml = YAML()
    # root_dir = '/home/ankur/MSR_Research_Home/Actor-Residual-Critic/logs/PlanarReachGoal1DenseFH-v0/exp-64/arc-f-max-rkl/2021_08_18_01_57_16'
    # config_file = 'variant_21139.yml'
    # ckpt_file = 'env_steps_9000.pt'
    root_dir = '/home/ankur/MSR_Research_Home/Actor-Residual-Critic/logs_ava/PlanarPushGoal1DenseFH-v0/exp-64/arc-f-max-rkl/2021_08_19_01_06_32'
    config_file = 'variant_44973.yml'
    ckpt_file = 'env_steps_26000.pt'
    v = yaml.load(open(os.path.join(root_dir,config_file)))

    # common parameters
    env_name = v['env']['env_name']
    env_T = v['env']['T']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['irl']['expert_episodes']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    print('Device is', device)
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()
    
    # assumptions
    assert v['obj'] in ['f-max-rkl', 'arc-f-max-rkl', 'gail', 'arc-gail', 'fairl', 'arc-fairl', 'airl', 'arc-airl', 'naive-diff-gail', 'naive-diff-f-max-rkl'] # approximate [RKL, JSD, FKL, RKL]

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    
    if v['adv_irl']['normalize']:
        expert_samples_ = expert_trajs.copy().reshape(-1, len(state_indices))
        obs_mean, obs_std = expert_samples_.mean(0), expert_samples_.std(0)
        obs_std[obs_std == 0.0] = 1.0 # avoid constant distribution
        expert_samples = (expert_samples - obs_mean) / obs_std # normalize expert data
        print('obs_mean, obs_std', obs_mean, obs_std)
        env_fn = lambda: gym.make(env_name, obs_mean=obs_mean, obs_std=obs_std)
    
    if v['obj'] in ['arc-f-max-rkl', 'arc-gail', 'arc-airl', 'arc-fairl', 'naive-diff-gail', 'naive-diff-f-max-rkl']:
        agent = SARC(env_fn, None, 
            steps_per_epoch=v['env']['T'],
            max_ep_len=v['env']['T'],
            seed=seed,
            reward_state_indices=state_indices,
            device=device,
            objective=v['obj'],
            reward_scale=v['adv_irl']['reward_scale'],
            **v['sac']
        )
    else:
        agent = SAC(env_fn, None, 
            steps_per_epoch=v['env']['T'],
            max_ep_len=v['env']['T'],
            seed=seed,
            reward_state_indices=state_indices,
            device=device,
            **v['sac']
        )

    agent.test_fn = agent.test_agent_ori_env
    agent.ac.load_state_dict(torch.load(os.path.join(root_dir,'agent',ckpt_file)))
    policy = agent.get_action
    _,_,returns = evaluate_policy(policy, gym_env, 10, deterministic=True)
    print(returns)