import gym, time
from icecream import ic
import numpy as np

env = gym.make('PlanarReachGoal1Dense-v1')

obs = env.reset()

def p_controller(obs, gain=10):
    action = np.concatenate((gain*obs,np.zeros(1)))
    return action

for episode in range(10):
    ic(episode)
    obs = env.reset()
    done = False
    for t in range(20):
        # a = env.action_space.sample()
        ic(obs)
        a = p_controller(obs)
        obs,rew,done,info = env.step(a)
        # print(np.round(obs,2))
        print(rew)
        env.render()
        time.sleep(0.1)