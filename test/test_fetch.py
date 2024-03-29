import gym, time
import envs
# from gym import envs
from icecream import ic
import numpy as np
import copy

# names = envs.registry.all()
# for name in names:
#     print(name)

env_name = 'PlanarPushGoal1DenseFH-v0'

env = gym.make(env_name, T=50)


# print(env.__dict__)
# obs = env.reset()

# def p_controller(obs, gain=30, max_val=1):
#     action = np.clip(gain*obs, -max_val, max_val)
#     return action

r_total = 0
for episode in range(1):
    ic(episode)
    obs = env.reset()
    done = False

    # gripper_pos = env.initial_gripper_xpos[:2]
    # ---------- set object position ---------- #
    # object_qpos = env.env.sim.data.get_joint_qpos('object0:joint')
    # object_qpos[:2] = gripper_pos + [0.1,0.15] + np.random.normal(scale=0.01, size=2)
    # env.env.sim.data.set_joint_qpos('object0:joint', object_qpos)
    
    # # ---------- set goal position ---------- #
    # goal_pos_rel = np.array([0.1,-0.15]) + np.random.normal(scale=0.01, size=2)
    # goal_pos_rel_3d = np.concatenate((goal_pos_rel, [0]))
    # env.env.goal = env.initial_gripper_xpos[:3] + goal_pos_rel_3d
    t = 0
    obs_prev = copy.deepcopy(obs)
    env.render()
    obs_arr = np.zeros((20,env.observation_space.shape[0]))
    # while not done:
    while t<20:
        print(t)
        a = env.action_space.sample()
        # a[:] = 0
        # if t<5:
        if env_name == 'PlanarReachGoal1DenseFH-v0':
            a[0] = 0.5
            a[1] = -0.5
        elif env_name == 'PlanarPushGoal1DenseFH-v0':
            a[0] = 0
            a[1] = -0.5
# # ---------- getting env name ---------- #
# # env_dict = gym.envs.registry.all()
# # for k in env_dict:
# # 	print(k)
# # EnvSpec(FetchSlide-v1)
# # EnvSpec(FetchPickAndPlace-v1)
# # EnvSpec(FetchReach-v1)
# # EnvSpec(FetchPush-v1)
# # EnvSpec(FetchSlideDense-v1)
# # EnvSpec(FetchPickAndPlaceDense-v1)
# # EnvSpec(FetchReachDense-v1)
# # EnvSpec(FetchPushDense-v1)
#         a = p_controller(obs)
#         # ic(a.round(2),obs.round(2))
        # input()
        obs,rew,done,info = env.step(a)
        obs_arr[t,:] = obs
        # diff = obs-obs_prev
        # obs_prev = copy.deepcopy(obs)
        # ic((diff*100).round(2))
        ic(obs)
        # print(env.T)
#         r_total += rew
#         # print(np.round(obs,2))
#         # print(rew)
        # ic(obs.round(2))
        # obs = obs['observation']
        # ic(obs[6:9], obs[11:14])
        env.render()
        time.sleep(0.1)
        t+=1
    print(obs_arr)
    np.savetxt(f'test/observation_space_{env_name}.csv', obs_arr, delimiter=',')
        
# print(r_total/10)
