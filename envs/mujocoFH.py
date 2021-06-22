# Fixed Horizon wrapper of mujoco environments
import gym, gym_vecenv
import numpy as np

import sys, os
sys.path.append(os.path.join(sys.path[0],'../mape'))
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import time

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
