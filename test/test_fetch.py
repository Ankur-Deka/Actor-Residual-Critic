import gym
from icecream import ic

env = gym.make('FetchReach-v1')
ic(env.state_space(), env.action_space())



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