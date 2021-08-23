import os



exp_list = \
[
{'env_name': 'PlanarPushGoal1DenseFH-v0',
'plot_name': 'FetchPush',
'exp_runs': {
    'arc-gail': ['2021_08_22_18_55_57',
                 '2021_08_22_19_18_14',
                 '2021_08_22_19_18_16',
                 '2021_08_22_19_18_18',
                 '2021_08_22_19_18_20'],

    'gail': ['2021_08_22_18_56_50',
             '2021_08_22_19_17_24',
             '2021_08_22_19_17_26',
             '2021_08_22_19_17_28',
             '2021_08_22_19_17_30'],

    'arc-f-max-rkl': ['2021_08_22_18_55_55',
                      '2021_08_22_19_18_06',
                      '2021_08_22_19_18_08',
                      '2021_08_22_19_18_10',
                      '2021_08_22_19_18_12'],

    'f-max-rkl': ['2021_08_22_18_56_48',
                  '2021_08_22_19_17_16',
                  '2021_08_22_19_17_18',
                  '2021_08_22_19_17_20',
                  '2021_08_22_19_17_22'],
    },
'max_steps': 20000}
]

expert_episodes = 64

def run_command(command):
    # print(command)
    os.system(command)

for exp in exp_list:
    env_name = exp['env_name']
    max_steps = exp['max_steps']
    for algo_name, algo_runs in exp['exp_runs'].items():
        for run in algo_runs:
            # make directory
            command = f'mkdir -p logs_ava/{env_name}/exp-{expert_episodes}/{algo_name}/{run}/agent'
            run_command(command)

            command_prefix = f'scp ankur@ava.ri.cmu.edu:MSR_Research/Actor-Residual-Critic/logs/{env_name}/exp-{expert_episodes}/{algo_name}/{run}'
            command_postfix = f'logs_ava/{env_name}/exp-{expert_episodes}/{algo_name}/{run}'

            # policy
            command = f'{command_prefix}/agent/env_steps_{max_steps}.pt {command_postfix}/agent/'
            run_command(command)

            # progress.csv
            command = f'{command_prefix}/progress.csv {command_postfix}/'
            run_command(command)

            # yml
            command = f'{command_prefix}/*.yml {command_postfix}/'
            run_command(command)         
# scp ankur@ava.ri.cmu.edu:MSR_Research/Actor-Residual-Critic/