<<<<<<< HEAD
import gym
print(gym)
=======
import gym, time
import envs
>>>>>>> 791b079a3610b2968270ceb8f8ef9d95b5de08bf
from icecream import ic
import numpy as np

<<<<<<< HEAD
env = gym.make('FetchReach-v1')
ic(env.__dict__)
#ic(env.state_space(), env.action_space())
=======
env = gym.make('PlanarReachGoal1DenseFH-v0')
print(env.__dict__)
obs = env.reset()
>>>>>>> 791b079a3610b2968270ceb8f8ef9d95b5de08bf

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

<<<<<<< HEAD
# ---------- getting env name ---------- #
# env_dict = gym.envs.registry.all()
# for k in env_dict:
# 	print(k)
# EnvSpec(FetchSlide-v1)
# EnvSpec(FetchPickAndPlace-v1)
# EnvSpec(FetchReach-v1)
# EnvSpec(FetchPush-v1)
# EnvSpec(FetchSlideDense-v1)
# EnvSpec(FetchPickAndPlaceDense-v1)
# EnvSpec(FetchReachDense-v1)
# EnvSpec(FetchPushDense-v1)
=======
        a = p_controller(obs)
        # ic(a.round(2),obs.round(2))
        obs,rew,done,info = env.step(a)
        r_total += rew
        # print(np.round(obs,2))
        # print(rew)
        env.render()
        # time.sleep(0.1)
print(r_total/10)
>>>>>>> 791b079a3610b2968270ceb8f8ef9d95b5de08bf
