# ARC - Actor Residual Critic for Adversarial Imitation Learning
Authors: Ankur Deka, Changliu Liu, Katia Sycara

## Installation

- PyTorch 1.5+
- OpenAI Gym (0.15+)
  `pip install gym[box2d]` - required for lunar lander
- [MuJoCo](https://www.roboti.us/license.html)
- `pip install sklearn seaborn ruamel.yaml`
- `pip3 install pybullet --upgrade --user` for pybullet for InvertedPendulum (continuous) 
- Download expert data that are used in our paper from [Google Drive](https://drive.google.com/drive/folders/1HJBV0HRi3B0KcbRAX5BOudkEYKj8ts4Q?usp=sharing) as `expert_data/` folder
  - `states/`: expert state trajectories for each environment. We obtain two sets of state trajectories for our method/MaxEntIRL/f-MAX (`*.pt`) and AIRL (`*_airl.pt`), respectively.
  - `actions/`: expert action trajectories for each environment for AIRL (`*_airl.pt`)
  - `meta/`: meta information including expert reward curves through training


## File Structure
- ARC (Our method): `firl/`
- Baselines (f-MAX, BC): `baselines/`
- SAC agent: `common/`
- Environments: `envs/`
- Configurations: `configs/`

## Instructions
- All the experiments are to be run under the root folder. 
- Before starting experiments, please `export PYTHONPATH=${PWD}:$PYTHONPATH` for env variable. 
- We use yaml files in `configs/` for experimental configurations, please change `obj` value (in the first line) for each method, here is the list of `obj` values:
    -  arc-f-max-rkl
    -  f-max-rkl
    -  naive-diff-f-max-rkl
    -  arc-gail
    -  gail
    -  naive-diff-gail
    -  bc
- Please keep all the other values in yaml files unchanged to reproduce the results in our paper.
- After running, you will see the training logs in `logs/` folder.

## Experiments
### Expert Data

First, make sure that you have downloaded expert data into `expert_data/`.  [Data is available here](https://drive.google.com/drive/folders/1HJBV0HRi3B0KcbRAX5BOudkEYKj8ts4Q?usp=sharing).
*Otherwise*, you can generate expert data by training expert policy:

```bash
python common/train_expert.py configs/samples/experts/{file}.yml # env is in {hopper, walker2d, halfcheetah, ant}
```

For standard AIL algorithms (f-max-rkl and gail) on walker2d, halfcheetah, ant, use standard_ail.yml. Use standard_ail_halfcheetah for halfcheetah.

For ARC aided AIL (ard-f-max-rkl and arc-gail) on walker2d, halfcheetah, ant use arc_ail.yml. Use arc_ail_halfcheetah.yml for halfcheetah.


## References
Standard AIL implementations are borrowed from [f-IRL repository](https://github.com/twni2016/f-IRL) and we developed ARC aided AIL on top of them. In turn, they used the following references for parts of the code:
- [AIRL](https://github.com/justinjfu/inverse_rl) in part of `envs/` 
- [f-MAX](https://github.com/KamyarGh/rl_swiss/blob/master/run_scripts/adv_smm_exp_script.py) in part of `baselines/`
- [SAC](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac) in part of `common/sac`
- [NPEET](https://github.com/gregversteeg/NPEET) in part of `utils/it_estimator.py`

We also use the expert data from [f-IRL repository](https://github.com/twni2016/f-IRL).