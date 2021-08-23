from matplotlib import pyplot as plt
import matplotlib
font = {'family' : 'serif',
        # 'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
from os import path as osp
from itertools import zip_longest, product
import numpy as np
import argparse, sys, os
import pandas as pd
from scipy.ndimage import uniform_filter
# matplotlib.style.use('seaborn-whitegrid')
from icecream import ic

def unpack(s):
    return " ".join(map(str, s))

def remove_nan(raw_data):
    return raw_data[~np.isnan(raw_data)]

root_dir = 'logs_ava'
log_interval = 1e3

# --------------------------------------------------------------------

# list of dictionaries
# exp_list = \
# [
# {'env_name': 'AntFH-v0', # port 6008
# 'plot_name': 'Ant',
# 'exp_runs': {
#     # 'sgail': ['2020_12_27_08_16_49', '2020_12_27_08_17_31', '2020_12_28_01_34_59'], # wrong objective, gpu # first one is sort of bad
#     # 'gail': ['2020_12_27_21_07_49', '2020_12_27_21_08_30', '2020_12_27_21_08_51'] # gpu
#     'sgail': ['2021_04_19_16_13_55', '2021_04_19_16_13_57', '2021_04_19_16_13_59'],#, '2021_04_19_16_14_01'],
#     'gail': ['2021_05_02_15_13_48', '2021_05_02_15_14_01', '2021_05_02_15_14_15'],
#     'naive-diff-gail': ['2021_06_13_02_17_07', '2021_06_12_13_24_24', '2021_06_12_13_24_26'], # seed 0 left
#     'arc-f-max-rkl': ['2021_04_17_23_45_44', '2021_04_18_20_17_23', '2021_04_18_20_17_25'],
#     'f-max-rkl': ['2021_04_17_21_50_40', '2021_04_18_14_06_52', '2021_04_18_14_07_10'],
#     'naive-diff-f-max-rkl': ['2021_06_12_13_01_52', '2021_06_12_13_01_54', '2021_06_12_13_01_56'],
#     'bc': ['2021_06_12_12_10_23','2021_06_12_12_10_25','2021_06_12_12_10_27']
#     },
# 'max_steps': 3e6
# },

# {
# 'env_name': 'Walker2dFH-v0', #port 6006
# 'plot_name': 'Walker2d',
# 'exp_runs': {
#     # 'sgail': ['2020_12_20_03_37_19', '2020_12_27_08_07_09', '2020_12_27_08_07_45'], # wrong objective
#     # 'gail': ['2020_12_27_21_02_52', '2020_12_27_21_03_47', '2020_12_27_21_04_01'], # GPU
#     'sgail': ['2021_04_19_16_13_45', '2021_04_19_16_13_47', '2021_04_19_16_13_49'],
#     'gail': ['2021_05_02_15_11_08', '2021_05_02_15_11_32', '2021_05_02_15_12_28'],
#     'naive-diff-gail': ['2021_06_12_13_24_12', '2021_06_12_13_24_12', '2021_06_12_13_24_16'],
#     'f-max-rkl': ['2021_04_18_00_47_07', '2021_04_18_12_24_12', '2021_04_18_12_24_30'], # checked
#     'arc-f-max-rkl': ['2021_04_18_20_17_11', '2021_04_18_20_17_13', '2021_04_18_20_17_15'], #
#     'naive-diff-f-max-rkl': ['2021_06_12_13_01_40', '2021_06_12_13_01_42', '2021_06_12_13_01_44'],
#     'bc': ['2021_06_12_12_10_11', '2021_06_12_12_10_13', '2021_06_12_12_10_15'] 
#     },
# 'max_steps': 5e6
# },

# {
# 'env_name': 'HalfCheetahFH-v0', # port 6009
# 'plot_name': 'HalfCheetah',
# 'exp_runs': {
#     # 'sgail': ['2021_01_04_00_28_53', '2021_01_04_00_29_33'], #'2021_01_04_00_29_08'   # wrong objective (reward func) in ssac.py and gpu
#     # 'gail': ['2021_01_04_00_31_58', '2021_01_04_00_32_46', '2021_01_04_10_46_33']     # gpu
#     'sgail': ['2021_05_04_10_29_54', '2021_05_03_13_15_56', '2021_05_03_13_15_59'],     # '2021_05_03_13_15_55'
#     'gail': ['2021_05_03_13_18_12', '2021_05_03_13_18_14', '2021_05_03_13_18_16'],
#     'naive-diff-gail': ['2021_06_12_13_24_28', '2021_06_12_13_24_31', '2021_06_12_13_24_32'],
#     'arc-f-max-rkl': ['2021_05_03_13_26_42', '2021_05_03_13_26_44' ,'2021_05_03_13_26_46'],
#     'f-max-rkl': ['2021_05_03_13_24_45', '2021_05_03_13_24_47', '2021_05_03_13_24_49'],
#     'naive-diff-f-max-rkl': ['2021_06_12_13_01_58', '2021_06_12_13_02_00', '2021_06_12_13_02_02'],
#     'bc': ['2021_06_12_12_10_29', '2021_06_12_12_10_31', '2021_06_12_12_10_33']
# },
# 'max_steps': 3e6
# },

# {
# 'env_name': 'HopperFH-v0', # port 6007
# 'plot_name': 'Hopper',
# 'exp_runs': {
#     # 'sgail': ['   ', '2020_12_27_08_13_05', '2021_01_03_02_51_47'], # wrong objective, gpu
#     # 'gail': ['2020_12_27_21_05_48', '2020_12_27_21_06_14', '2020_12_27_21_06_35'], gpu
#     'sgail': ['2021_04_19_16_13_51', '2021_04_19_16_13_53'],
#     'gail': ['2021_05_02_16_05_49', '2021_05_02_16_06_08', '2021_05_02_16_06_25'],
#     'arc-f-max-rkl': ['2021_04_18_20_17_17', '2021_04_18_20_17_19', '2021_04_18_20_17_21'],
#     'f-max-rkl': ['2021_04_17_21_50_25', '2021_04_18_12_25_07', '2021_04_18_12_25_47'],
#     'naive-diff-f-max-rkl': ['2021_06_12_13_01_46','2021_06_12_13_01_48','2021_06_12_13_01_50'],
#     'naive-diff-gail': ['2021_06_12_13_24_18', '2021_06_12_13_24_20','2021_06_12_13_24_22'],
#     'bc': ['2021_06_12_12_10_17', '2021_06_12_12_10_19', '2021_06_12_12_10_21']
# },
# 'max_steps': 1e6
# }

# ]


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


# -----------------------------------------------------------------------
algo_ids = [
            # 'expert',
            # 'bc', 
            'arc-f-max-rkl', 
            'arc-gail',
            'f-max-rkl',
            'gail', 
            # 'naive-diff-f-max-rkl', 
            # 'naive-diff-gail'
            ]

algo_names = {
    'arc-gail': 'ARC-GAIL (Our)',
    'gail': 'GAIL',
    'arc-f-max-rkl': r'ARC-$f$-Max-RKL (Our)',
    'f-max-rkl': r'$f$-Max-RKL',
    'naive-diff-gail': 'Naive-Diff GAIL',
    'naive-diff-f-max-rkl': r'Naive-Diff $f$-Max-RKL',
    'bc': 'BC',
    'expert': 'Expert'
}
num_trajs = 64
colors = {
    'gail': 'orange',
    'arc-gail': 'red',
    'arc-f-max-rkl': 'blue',
    'f-max-rkl': 'green',
    'naive-diff-gail': 'purple',
    'naive-diff-f-max-rkl': 'pink',
    'expert': 'black',
    'bc': 'gray'
}

means = []
stds = []

for exp in exp_list:
    env_name = exp['env_name']
    plot_name = exp['plot_name']
    exp_runs = exp['exp_runs']
    max_steps = exp['max_steps']

    show_legend = False
    max_steps = int(max_steps//log_interval+1)

    exp_path = os.path.join(root_dir, env_name, 'exp-{}'.format(num_trajs))

    expert_file = open(f'expert_data/meta/{env_name}.txt', 'r')
    first_line = expert_file.readline().rstrip().split()
    loc = first_line.index('Avg:')
    expert_return = float(first_line[loc+1][:-1])
    metric = 'Real Det Return'
    x_axis = 'Running Env Steps'

    plt.figure()
    mean = []
    std = []

    for algo in algo_ids:
        if algo == 'expert':
            continue
        runs = exp_runs[algo]
        # print(algo)
        y_data = []
        best_run = None
        best_return = -np.inf
        for run in runs:
            path = os.path.join(exp_path,algo,run, 'progress.csv')
            # print(path)

            if algo=='bc':
                df = pd.read_csv(path).head(n=100)
                data = df[metric].values[-1]
                print(data)
                data = [data,data]
            else:
                df = pd.read_csv(path).head(n=max_steps)
                x = remove_nan(df.loc[:, x_axis].values)
                data = remove_nan(df.loc[:, metric].values) # to numpy array
            # print(raw_data.shape)
            # data = uniform_filter(raw_data, 20)
            final_return = data[-1]
            if final_return>best_return:
                best_return = final_return
                best_run = run
            y_data.append(data)

        ic(algo, best_return, best_run)
        y_mean = np.mean(y_data, axis=0)
        y_std = np.std(y_data, axis=0)
        mean.append(y_mean[-1])
        std.append(y_std[-1])
        

        if algo=='bc':
            plt.axhline(y=y_mean[-1], linestyle='--', color=colors['bc'], label='bc')
        else:
            plt.plot(x, y_mean, color=colors[algo], label=algo_names[algo])
            plt.fill_between(x,y_mean-y_std,y_mean+y_std, color=colors[algo], alpha=0.5)
        # plt.text(y[-1], data[-1], f'{div}')
    means.append(mean)
    stds.append(std)

    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    # plt.rcParams.update({'font.size': 12})
    # plt.tick_params(axis='both', which='major', labelsize=20)
    # plt.tick_params(axis='both', which='minor', labelsize=20)
    # plt.axhline(y=expert_return, linestyle='--', color=colors['expert'], label='expert')
    plt.grid()
    plt.title(plot_name, fontsize=28)
    if show_legend:
        plt.legend()
    plt.tight_layout()
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(os.path.join('results', '{}.pdf'.format(env_name)))
    # plt.show()    


# generate legend image
fig = plt.figure()
fig_legend = plt.figure(figsize=(2, 1.25))
ax = fig.add_subplot(111)
lines = []
labels = []
for alg_id in algo_ids:
    name = algo_names[alg_id]
    color = colors[alg_id]
    style = '--' if alg_id in ['expert', 'bc'] else '-'
    lines.append(ax.plot([1,2],[1,2],linestyle=style,c=color)[0])
    labels.append(name)

# fig_legend.legend(lines, labels, loc='center', frameon=False, ncol=4)
# plt.show()
# fig_legend.show()



means = np.array(means).round(2)
stds = np.array(stds).round(2)
table_string = ''
for algo in range(4):
    alg_id = algo_ids[algo]
    table_string += f'{algo_names[alg_id]} & '
    for env in range(1):
        mean = means[env, algo]
        std = stds[env,algo]
        if alg_id in ['arc-f-max-rkl', 'arc-gail']:
            table_string += f'\\textbf{ {mean} } $\\pm$ {std} & '
        else:
            table_string += f'{mean} $\\pm$ {std} & '
    table_string = table_string[:-1] + '\\\\ \n'

with open('results/table_data.txt','a') as f:
    f.write(table_string)










