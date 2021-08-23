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

import pandas as pd
from icecream import ic
import cv2
import matplotlib.pyplot as plt
from test.save_expert_policy import p_controller, hand_coded_push_policy

def unpack(s):
    return " ".join(map(str, s))

def remove_nan(raw_data):
    return raw_data[~np.isnan(raw_data)]

algo_names = {
    'arc-gail': 'ARC-GAIL (Our)',
    'gail': 'GAIL',
    'arc-f-max-rkl': r'ARC-$f$-Max-RKL (Our)',
    'f-max-rkl': r'$f$-Max-RKL',
    'naive-diff-gail': 'Naive-Diff GAIL',
    'naive-diff-f-max-rkl': r'Naive-Diff $f$-Max-RKL',
    'bc': 'BC',
    'expert': 'Expert'
}
num_trajs = 64
colors = {
    'gail': 'orange',
    'arc-gail': 'red',
    'arc-f-max-rkl': 'blue',
    'f-max-rkl': 'green',
    'naive-diff-gail': 'purple',
    'naive-diff-f-max-rkl': 'pink',
    'expert': 'black',
    'bc': 'gray'
}

expert_actions_reach = np.array([[[ 1.0000, -1.0000],
                                 [ 1.0000, -1.0000],
                                 [ 0.8903, -0.8510],
                                 [ 0.6117, -0.5639],
                                 [ 0.4174, -0.3705],
                                 [ 0.2873, -0.2447],
                                 [ 0.1994, -0.1622],
                                 [ 0.1399, -0.1080],
                                 [ 0.0992, -0.0722],
                                 [ 0.0711, -0.0485],
                                 [ 0.0515, -0.0328],
                                 [ 0.0377, -0.0222],
                                 [ 0.0277, -0.0152],
                                 [ 0.0204, -0.0104],
                                 [ 0.0150, -0.0071],
                                 [ 0.0111, -0.0049],
                                 [ 0.0082, -0.0034],
                                 [ 0.0059, -0.0024],
                                 [ 0.0040, -0.0016],
                                 [ 0.0025, -0.0010]]])

expert_actions_push = np.array([[[ 4.4750e-04, -1.0000e+00],
                                 [ 3.2111e-03, -1.0000e+00],
                                 [ 5.2910e-03, -1.0000e+00],
                                 [ 1.0057e-02, -1.0000e+00],
                                 [ 1.3781e-02, -1.0000e+00],
                                 [ 2.1920e-02, -1.0000e+00],
                                 [ 2.7846e-02, -7.9720e-01],
                                 [ 3.0019e-02, -5.2911e-01],
                                 [ 2.9819e-02, -3.4840e-01],
                                 [ 2.8730e-02, -2.2994e-01],
                                 [ 2.2977e-02, -1.5195e-01],
                                 [ 1.2726e-02, -1.0058e-01],
                                 [ 1.2386e-02, -6.6515e-02],
                                 [ 1.0870e-02, -4.4199e-02],
                                 [ 9.2363e-03, -2.9628e-02],
                                 [ 6.4296e-03, -1.9915e-02],
                                 [ 5.6954e-03, -1.3602e-02],
                                 [ 4.6063e-03, -9.4594e-03],
                                 [ 3.5502e-03, -6.6748e-03],
                                 [ 2.5937e-03, -4.7025e-03],
                                 [ 1.8528e-03, -3.3324e-03],
                                 [ 1.1187e-03, -2.3441e-03],
                                 [ 6.5036e-04, -1.6144e-03],
                                 [ 1.9169e-04, -1.0741e-03],
                                 [-2.8606e-04, -6.6472e-04],
                                 [-4.2062e-04, -3.6182e-04],
                                 [-7.3215e-04, -1.2057e-04],
                                 [-8.1533e-04,  5.0317e-05],
                                 [-8.1676e-04,  1.7635e-04],
                                 [-9.2244e-04,  2.7028e-04]]])


def evaluate_policy(policy, env, n_episodes, deterministic=False,save_path=None):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    expert_states = torch.zeros((n_episodes, env_T, obs_dim)) # s1 to sT
    expert_actions = torch.zeros((n_episodes, env_T, act_dim)) # a0 to aT-1

    returns = []
    if not save_path is None:
        os.makedirs(save_path, exist_ok=True) 
    # t_max = 0 # max episode seen at evaluation

    for n in range(n_episodes):
        print(n)
        obs = env.reset()
        ret = 0
        t = 0
        done = False
        stored_terminal_rew = False
        # while not done and t<env_T:
        obs_size = 4 if env_name.startswith('PlanarPush') else 2
        obs_arr = np.zeros((env_T,obs_size))
        action_arr = np.zeros((env_T,2))
        # env.render()
        while t<env_T:
            # print(t)
            action = policy(obs, deterministic)
            print(obs, action)
            obs_arr[t,:] = obs
            action_arr[t,:] = action
            # action = np.array([1,0])
            obs, rew, done, _ = env.step(action) # NOTE: assume rew=0 after done=True for evaluation
            
            if done:
                rew = 0 if stored_terminal_rew else rew
                stored_terminal_rew = True
            expert_states[n, t, :] = torch.from_numpy(obs).clone()
            expert_actions[n, t, :] = torch.from_numpy(action).clone()
            ret += rew
            t += 1
            # frame = env.render(mode='rgb_array', width=1000, height=1000)
            # frame = frame[:,:,[2,1,0]]

            # print(frame)
            # cv2.imwrite(f'{save_path}/{env_name}_{algo}_{t}.png', frame)
            # input()
            # time.sleep(0.1)
        
        # np.savetxt('simulator_obs_fmax.csv', obs_arr.round(3), delimiter=',')
        # np.savetxt('simulator_action_fmax.csv', action_arr.round(3), delimiter=',')
        
            # t_max = max(t_max,t)
        returns.append(ret)

    # return expert_states[:,:t_max], expert_actions[:,:t_max], np.array(returns) # clip to max episode length seen at evaluation
    return expert_states, expert_actions, np.array(returns)

def get_observation_space_actions(policy, obs_arr, action_size):
    action_arr = np.zeros((20,action_size))
    for i,obs in enumerate(obs_arr):
        action_arr[i] = policy(obs,True)
    return action_arr

def get_expert_actions(obs_arr, env, env_name,action_size):
    action_arr = np.zeros((20,action_size))
    if env_name=='PlanarReachGoal1DenseFH-v0':
        for i,obs in enumerate(obs_arr):
            print(p_controller(obs))
            action_arr[i] = p_controller(obs)
    elif env_name == 'PlanarPushGoal1DenseFH-v0':
        for i,obs in enumerate(obs_arr):
            action_arr[i] = hand_coded_push_policy(env,obs,i)
    return action_arr

if __name__ == "__main__":
    yaml = YAML()
    # root_dir = '/home/ankur/MSR_Research_Home/Actor-Residual-Critic/logs/PlanarReachGoal1DenseFH-v0/exp-64/arc-f-max-rkl/2021_08_18_01_57_16'
    # config_file = 'variant_21139.yml'
    # ckpt_file = 'env_steps_9000.pt'

    # # ---------- arc-fmax
    # root_dir = '/home/ankur/MSR_Research_Home/Actor-Residual-Critic/logs_ava/PlanarPushGoal1DenseFH-v0/exp-64/arc-f-max-rkl/2021_08_19_01_06_32'
    # config_file = 'variant_44973.yml'
    # ckpt_file = 'env_steps_500000.pt'

    # # ---------- fmax
    # root_dir = '/home/ankur/MSR_Research_Home/Actor-Residual-Critic/logs_ava/PlanarPushGoal1DenseFH-v0/exp-64/f-max-rkl/2021_08_19_01_52_10'
    # config_file = 'variant_51349.yml'
    # ckpt_file = 'env_steps_500000.pt'

    # ---------- new friction values ---------- #
    # ---------- arc-fmax
    # root_dir = '/home/ankur/MSR_Research_Home/Actor-Residual-Critic/logs_ava/PlanarPushGoal1DenseFH-v0/exp-64/arc-f-max-rkl/2021_08_22_17_34_36'
    # config_file = 'variant_13588.yml'
    # ckpt_file = 'env_steps_20000.pt'

    # ---------- fmax
    # root_dir = '/home/ankur/MSR_Research_Home/Actor-Residual-Critic/logs_ava/PlanarPushGoal1DenseFH-v0/exp-64/f-max-rkl/2021_08_22_17_36_04'
    # config_file = 'variant_14127.yml'
    # ckpt_file = 'env_steps_20000.pt'

    # ---------- new friction, longer time horizon ----------
    # ---------- arc-fmax
    # root_dir = '/home/ankur/MSR_Research_Home/Actor-Residual-Critic/logs_ava/PlanarPushGoal1DenseFH-v0/exp-64/arc-f-max-rkl/2021_08_22_18_55_55'
    # config_file = 'variant_25493.yml'
    # ckpt_file = 'env_steps_20000.pt'
    
    # ---------- fmax
    # root_dir = '/home/ankur/MSR_Research_Home/Actor-Residual-Critic/logs_ava/PlanarPushGoal1DenseFH-v0/exp-64/f-max-rkl/2021_08_22_18_56_48'
    # config_file = 'variant_25962.yml'
    # ckpt_file = 'env_steps_20000.pt'

    exp_list = \
    [
    {'env_name': 'PlanarReachGoal1DenseFH-v0',
    'plot_name': 'FetchReach',
    'exp_runs': {
        'arc-gail': ['2021_08_23_02_21_20',
                     '2021_08_23_02_21_22',
                     '2021_08_23_02_21_24',
                     '2021_08_23_02_21_26',
                     '2021_08_23_02_21_28'],

        'gail': ['2021_08_23_02_23_12',
                 '2021_08_23_02_23_14',
                 '2021_08_23_02_23_16',
                 '2021_08_23_02_23_18',
                 '2021_08_23_02_23_20'],

        'arc-f-max-rkl': ['2021_08_23_02_21_10',
                          '2021_08_23_02_21_12',
                          '2021_08_23_02_21_14',
                          '2021_08_23_02_21_16',
                          '2021_08_23_02_21_18'],

        'f-max-rkl': ['2021_08_23_02_23_02',
                      '2021_08_23_02_23_04',
                      '2021_08_23_02_23_06',
                      '2021_08_23_02_23_08',
                      '2021_08_23_02_23_10'],
        },
    'max_steps': 25000},
    # {'env_name': 'PlanarPushGoal1DenseFH-v0',
    # 'plot_name': 'FetchPush',
    # 'exp_runs': {
    #     'arc-gail': ['2021_08_23_01_06_36',
    #                  '2021_08_23_01_06_38',
    #                  '2021_08_23_01_06_40',
    #                  '2021_08_23_01_06_42',
    #                  '2021_08_23_01_06_44'],

    #     'gail': ['2021_08_23_01_07_09',
    #              '2021_08_23_01_07_11',
    #              '2021_08_23_01_07_13',
    #              '2021_08_23_01_07_15',
    #              '2021_08_23_01_07_17'],

    #     'arc-f-max-rkl': ['2021_08_23_01_06_26',
    #                       '2021_08_23_01_06_28',
    #                       '2021_08_23_01_06_30',
    #                       '2021_08_23_01_06_32',
    #                       '2021_08_23_01_06_34'],

    #     'f-max-rkl': ['2021_08_23_01_06_58',
    #                   '2021_08_23_01_07_00',
    #                   '2021_08_23_01_07_02',
    #                   '2021_08_23_01_07_05',
    #                   '2021_08_23_01_07_06'],
    #     },
    # 'max_steps': 25000}
    ]
    root_dir = 'logs_ava'
    log_interval = 1e3
    num_trajs = 64
    metric = 'Real Det Return'
    x_axis = 'Running Env Steps'
    # -----------------------------------------------------------------------
    algo_ids = [
                # 'expert',
                # 'bc', 
                'arc-f-max-rkl', 
                'arc-gail',
                'f-max-rkl',
                'gail', 
                # 'naive-diff-f-max-rkl', 
                # 'naive-diff-gail'
                ]
    
    for exp in exp_list:
        env_name = exp['env_name']
        exp_runs = exp['exp_runs']
        max_steps = exp['max_steps']
        plot_name = exp['plot_name']

        max_steps = int(max_steps//log_interval+1)
        exp_path = os.path.join(root_dir, env_name, 'exp-{}'.format(num_trajs))
        print(exp_path)
        for algo in algo_ids:
            runs = exp_runs[algo]
            # print(algo)
            y_data = []
            best_run = None
            best_return = -np.inf
            for run in runs:
                path = os.path.join(exp_path,algo,run, 'progress.csv')
                if algo=='bc':
                    df = pd.read_csv(path).head(n=100)
                    data = df[metric].values[-1]
                    print(data)
                    data = [data,data]
                else:
                    df = pd.read_csv(path).head(n=max_steps)
                    x = remove_nan(df.loc[:, x_axis].values)
                    data = remove_nan(df.loc[:, metric].values) # to numpy array
                final_return = data[-1]
                if final_return>best_return:
                    best_return = final_return
                    best_run = run
            ic(algo, best_return, best_run)

            run_path = os.path.join(exp_path, algo, best_run)
            
            # ---------- get config_file ----------
            files = os.listdir(run_path)
            config_file = None
            for file in files:
                if file.endswith('.yml'):
                    config_file = file
                    break

            # ---------- get ckpt_file ----------
            ckpt_file = f'env_steps_{exp["max_steps"]}.pt'

            ic(run_path, config_file, ckpt_file)

            v = yaml.load(open(os.path.join(run_path,config_file)))
            # common parameters
            env_name = v['env']['env_name']
            env_T = v['env']['T']
            ic(env_T)
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
            agent.ac.load_state_dict(torch.load(os.path.join(run_path,'agent',ckpt_file)))
            policy = agent.get_action
            _,actions,returns = evaluate_policy(policy, gym_env, 1, deterministic=True, save_path=f'results/{env_name}/{algo}')
            # print(actions)
            # plt.plot(-actions[0,:,1], color=colors[algo], label=algo_names[algo], linewidth=3)
            # ---------- states vs actions
            observation_space_arr = np.loadtxt(f'test/observation_space_{env_name}.csv', delimiter=',')
            observation_space_actions =  get_observation_space_actions(policy,observation_space_arr, action_size)
            print(observation_space_actions)
            plt.plot(-observation_space_actions[:,1], color=colors[algo], label=algo_names[algo], linewidth=3)
            

            print(returns)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        # plt.margins(0,0)
        # plt.savefig(os.path.join('results', '{}.pdf'.format(env_name)))
        # if env_name == 'PlanarReachGoal1DenseFH-v0':
        #     expert_actions = expert_actions_reach
        # else:
        #     expert_actions = expert_actions_push
        # plt.plot(-expert_actions[0,:,1], color=colors['expert'], label='Expert', linestyle='--', linewidth=3)
        expert_actions = get_expert_actions(observation_space_arr, gym_env, env_name, action_size)
        ic(expert_actions)
        plt.plot(-expert_actions[:,1], color=colors['expert'], label='Expert', linestyle='--', linewidth=3)
        # plt.rcParams.update({'font.size': 12})
        # plt.tick_params(axis='both', which='major', labelsize=20)
        # plt.tick_params(axis='both', which='minor', labelsize=20)
        # plt.grid()
        # plt.title(plot_name, fontsize=28)
        # if show_legend:
        # plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join('results', f'state_action_plot_{env_name}.pdf'))
        plt.show()

        # ---------- state space vs action ----------


