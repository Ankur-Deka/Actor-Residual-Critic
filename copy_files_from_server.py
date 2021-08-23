import os



exp_list = \
[
{'env_name': 'PlanarPushGoal1DenseFH-v0',
'plot_name': 'FetchPush',
'exp_runs': {
    'arc-gail': ['2021_08_23_01_06_36',
                 '2021_08_23_01_06_38',
                 '2021_08_23_01_06_40',
                 '2021_08_23_01_06_42',
                 '2021_08_23_01_06_44'],

    'gail': ['2021_08_23_01_07_09',
             '2021_08_23_01_07_11',
             '2021_08_23_01_07_13',
             '2021_08_23_01_07_15',
             '2021_08_23_01_07_17'],

    'arc-f-max-rkl': ['2021_08_23_01_06_26',
                      '2021_08_23_01_06_28',
                      '2021_08_23_01_06_30',
                      '2021_08_23_01_06_32',
                      '2021_08_23_01_06_34'],

    'f-max-rkl': ['2021_08_23_01_06_58',
                  '2021_08_23_01_07_00',
                  '2021_08_23_01_07_02',
                  '2021_08_23_01_07_05',
                  '2021_08_23_01_07_06'],
    },
'max_steps': 25000}
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

            # config
            command = f'{command_prefix}/*.yml {command_postfix}/'
            run_command(command)         
# scp ankur@ava.ri.cmu.edu:MSR_Research/Actor-Residual-Critic/