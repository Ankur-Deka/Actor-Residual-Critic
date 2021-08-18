import gym, time
import envs
from icecream import ic
import numpy as np

env = gym.make('PlanarReachGoal1DenseFH-v0')
print(env.__dict__)
obs = env.reset()

def p_controller(obs, gain=30, max_val=1):
    action = np.clip(gain*obs, -max_val, max_val)
    return action

r_total = 0
for episode in range(10):
    ic(episode)
    obs = env.reset()
    done = False
    for t in range(20):
        a = env.action_space.sample()

        a = p_controller(obs)
        # ic(a.round(2),obs.round(2))
        obs,rew,done,info = env.step(a)
        r_total += rew
        # print(np.round(obs,2))
        # print(rew)
        env.render()
        # time.sleep(0.1)
print(r_total/10)