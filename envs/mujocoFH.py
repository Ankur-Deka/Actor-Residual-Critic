# Fixed Horizon wrapper of mujoco environments
import gym, gym_vecenv
from gym import spaces
import numpy as np


import sys, os
sys.path.append(os.path.join(sys.path[0],'../mape'))
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import time
from icecream import ic
def make_multiagent_env(env_id, num_agents, dist_threshold, arena_size, identity_size, num_steps=128, diff_reward=True, same_color = False, random_leader_name=False, goal_at_top=False, hide_goal=False, video_format='webm'):
    scenario = scenarios.load(env_id+".py").Scenario(num_agents=num_agents, dist_threshold=dist_threshold,
                                                     arena_size=arena_size, identity_size=identity_size, num_steps=num_steps, diff_reward=diff_reward, same_color = same_color, random_leader_name=random_leader_name, goal_at_top=goal_at_top, hide_goal=hide_goal)
    world = scenario.make_world()
    
    env = MultiAgentEnv(world=world, 
                        reset_callback=scenario.reset_world, 
                        reward_callback=scenario.reward, 
                        observation_callback=scenario.observation,
                        info_callback=scenario.info if hasattr(scenario, 'info') else None,
                        done_callback=scenario.done,
                        discrete_action=False,
                        cam_range=arena_size,
                        num_steps=num_steps,
                        video_format=video_format)
    return env

class MujocoFH(gym.Env):
    def __init__(self, env_name, T=1000, r=None, obs_mean=None, obs_std=None, seed=1):
        if env_name.startswith('simple_turn'):
            self.env = make_multiagent_env(env_name,1,0.1,1,1)
        else:
            self.env = gym.make(env_name)
        self.T = T
        self.r = r
        assert (obs_mean is None and obs_std is None) or (obs_mean is not None and obs_std is not None)
        self.obs_mean, self.obs_std = obs_mean, obs_std

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.seed(seed)
        
    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        self.t = 0
        self.terminated = False
        self.terminal_state = None

        self.obs = self.env.reset()
        self.obs = self.normalize_obs(self.obs)
        return self.obs.copy()
    
    def step(self, action):
        self.t += 1

        if self.terminated:
            return self.terminal_state, 0, self.t == self.T, True
        else:
            prev_obs = self.obs.copy()
            self.obs, r, done, info = self.env.step(action)
            self.obs = self.normalize_obs(self.obs)
            
            if self.r is not None:  # from irl model
                r = self.r(prev_obs)

            if done:
                self.terminated = True
                self.terminal_state = self.obs
            
            return self.obs.copy(), r, done, done
    
    def normalize_obs(self, obs):
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
        return obs

    def render(self):
        self.env.render()


class FetchFH(MujocoFH):
    def __init__(self,*args,**kwargs):
        super(FetchFH,self).__init__(*args,**kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[2], dtype='float32')  
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')
        self.reset()

    # use fixed goal
    def extract_relative_goal_pos(self, obs):    
        grip_pos = obs['observation'][:2] # x.y,z position
        goal_pos_rel = self.goal - grip_pos
        return goal_pos_rel

    def reset(self):
        self.t = 0
        self.terminated = False
        self.terminal_state = None
        obs = self.env.reset()
        # self.obs = self.normalize_obs(self.obs)
        goal_pos_rel = np.array([0.15,0]) + np.random.normal(scale=0.01, size=2)
        goal_pos_rel_3d = np.concatenate((goal_pos_rel, [0]))
        # ic(self.env.goal)
        self.env.env.goal = self.env.initial_gripper_xpos[:3] + goal_pos_rel_3d
        self.goal = self.env.env.goal[:2]
        self.obs = goal_pos_rel
        return self.obs.copy()

    def step(self, action):
        self.t += 1

        if self.terminated:
            return self.terminal_state, 0, self.t == self.T, True
        else:
            prev_obs = self.obs.copy()
            action = np.concatenate((action,np.zeros(2)))
            obs, r, done, info = self.env.step(action)
            self.obs = self.extract_relative_goal_pos(obs)
            # self.obs = self.normalize_obs(self.obs)
            
            if self.r is not None:  # from irl model
                r = self.r(prev_obs)

            if done:
                self.terminated = True
                self.terminal_state = self.obs
            
            return self.obs.copy(), r, done, done
        # print(self.obs)
        return self.obs.copy(),reward,done,info    



class PushFH(MujocoFH):
    def __init__(self,*args,**kwargs):
        super(PushFH,self).__init__(*args,**kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[7], dtype='float32')  
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')
        self.reset()

    # use fixed goal
    def extract_relative_goal_pos(self, obs):    
        grip_pos = obs['observation'][:2] # x.y,z position
        goal_pos_rel = self.goal - grip_pos
        return goal_pos_rel

    def extract_object_rot(self, obs):
        object_rot = obs['observation'][11:14]
        return object_rot.copy()

    def extract_relative_object_pos(self, obs):
        object_pos_rel = obs['observation'][6:8]
        return object_pos_rel

    def reset(self):
        self.t = 0
        self.terminated = False
        self.terminal_state = None
        obs = self.env.reset()
        
        gripper_pos = self.env.initial_gripper_xpos[:2]
        # ---------- set object position ---------- #
        object_qpos = self.env.env.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:2] = gripper_pos + [0.1,0.15] + np.random.normal(scale=0.01, size=2)
        self.env.env.sim.data.set_joint_qpos('object0:joint', object_qpos)
        
        # ---------- set goal position ---------- #
        goal_pos_rel = np.array([0.1,-0.15]) + np.random.normal(scale=0.01, size=2)
        goal_pos_rel_3d = np.concatenate((goal_pos_rel, [0]))
        self.env.env.goal = self.env.initial_gripper_xpos[:3] + goal_pos_rel_3d
        self.goal = self.env.env.goal[:2]

        # ---------- set observation (goal_pos_rel, object_rot) ---------- #
        object_rot = self.extract_object_rot(obs)
        self.obs = np.concatenate((goal_pos_rel, object_rot))
        return self.obs.copy()

    def step(self, action):
        self.t += 1

        if self.terminated:
            return self.terminal_state, 0, self.t == self.T, True
        else:
            prev_obs = self.obs.copy()
            action = np.concatenate((action,np.zeros(2)))
            obs, r, done, info = self.env.step(action)
           
            goal_pos_rel = self.extract_relative_goal_pos(obs) 
            
            object_pos_rel = self.extract_relative_object_pos(obs)
            object_rot = self.extract_object_rot(obs)
            self.obs = np.concatenate((object_pos_rel, object_rot, goal_pos_rel))
            # self.obs = self.normalize_obs(self.obs)
            
            if self.r is not None:  # from irl model
                r = self.r(prev_obs)

            if done:
                self.terminated = True
                self.terminal_state = self.obs
            
            return self.obs.copy(), r, done, done
        # print(self.obs)
        return self.obs.copy(),reward,done,info    

