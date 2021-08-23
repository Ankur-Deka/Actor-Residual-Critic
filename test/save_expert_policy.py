import gym, time
import envs
from icecream import ic
import numpy as np
import sys, os, time
from ruamel.yaml import YAML
from utils import system
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time



def p_controller(obs, gain=10, max_val=1):
    action = np.clip(gain*obs, -max_val, max_val)
    return action

# env = gym.make(env_name)
# print(env.__dict__)
# obs = env.reset()
# r_total = 0
# for episode in range(10):
#     ic(episode)
#     obs = env.reset()
#     done = False
#     for t in range(20):
#         a = env.action_space.sample()

#         a = p_controller(obs)
#         # ic(a.round(2),obs.round(2))
#         obs,rew,done,info = env.step(a)
#         r_total += rew
#         # print(np.round(obs,2))
#         # print(rew)
#         env.render()
#         # time.sleep(0.1)
# print(r_total/10)

def hand_coded_push_policy(env, obs, t, gain=10, max_val=1):
    origin_pos = env.env.initial_gripper_xpos[:2]
    object_initial_pose_absolute = env.object_initial_pos
    object_initial_pose = object_initial_pose_absolute - origin_pos
    goal_absolute = env.goal
    goal = goal_absolute - origin_pos
    waypoints = [[object_initial_pose[0],goal[1]+0.03]]
    waypoint_timesteps = [100]
    i = 0
    while(waypoint_timesteps[i]<=t):
        i+=1
    cur_waypoint = waypoints[i]
    grip_pos_absolute = env.env.env.sim.data.get_site_xpos('robot0:grip')[:2]

    gripper_pos = grip_pos_absolute - origin_pos
    return p_controller(cur_waypoint-gripper_pos, gain, max_val)


def evaluate_policy(policy, env, n_episodes):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    expert_states = torch.zeros((n_episodes, env_T, obs_dim)) # s1 to sT
    expert_actions = torch.zeros((n_episodes, env_T, act_dim)) # a0 to aT-1

    returns = []

    # t_max = 0 # max episode seen at evaluation
    for n in range(n_episodes):
        obs = env.reset()
        ret = 0
        t = 0
        done = False
        stored_terminal_rew = False
        print(env.goal)
        # while not done and t<env_T:
        while t<env_T:
            # ---------- reach task
            # action = policy(obs)

            # ---------- push task
            action = policy(env,obs,t)
            # action[:] = 0
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
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name = v['env']['env_name']
    env_T = v['env']['T']
    # env_T = 1000 if env_name in ['InvertedPendulum-v2','LunarLanderContinuous-v2'] else v['env']['T'] 
    seed = v['seed']

    # system: device, threads, seed, pid
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    if env_name in ['InvertedPendulum-v2','LunarLanderContinuous-v2']:
        EnvCls = lambda : gym.make(env_name)
    else: 
        EnvCls = lambda : gym.make(env_name, T=env_T)
    
    # ---------- reach task
    # policy = p_controller

    # ---------- push task
    policy = hand_coded_push_policy
    env = EnvCls()
    env.seed(seed+1)

    
    # log_txt = open(f"expert_data/meta/{env_name}_{seed}.txt", 'w')
    
    # sns.violinplot(data=expert_returns, ax=axs[1])
    # axs[1].set_title("violin plot of expert return")
    # plt.savefig(os.path.join(f'expert_data/meta/{env_name}_{seed}.png')) 

    expert_states_det, expert_actions_det, expert_returns = evaluate_policy(policy, env, 1)
    print(expert_actions_det)
    # return_info = f'Expert(Det) Return Avg: {expert_returns.mean():.2f}, std: {expert_returns.std():.2f}'
    # print(return_info)
    # log_txt.write(return_info + '\n')
    # log_txt.write(repr(expert_returns)+'\n')
    # log_txt.write(repr(v))
    # os.makedirs('expert_data/states/', exist_ok=True)
    # os.makedirs('expert_data/actions/', exist_ok=True)
    # torch.save(expert_states_det, f'expert_data/states/{env_name}_det.pt')
    # torch.save(expert_states_det, f'expert_data/states/{env_name}_airl.pt')

    # torch.save(expert_actions_det, f'expert_data/actions/{env_name}_det.pt')
    # torch.save(expert_actions_det, f'expert_data/actions/{env_name}_airl.pt')
