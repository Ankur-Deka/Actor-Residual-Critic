import os
import multiprocessing
import time

fPath = 'configs/samples/agents/arc_ail_robot.yml'
file = open(fPath)
lines = file.readlines()
file.close()

seeds = [0,1,2,3,4]
# seeds = [0]
# envs = ['Walker2dFH-v0', 'HopperFH-v0', 'AntFH-v0']
algos = ['arc-f-max-rkl', 'arc-gail']
envs = ['PlanarReachGoal1DenseFH-v0']
process_list = []

def run_experiment():
    os.system(f'python baselines/main_samples.py {fPath}')

for algo in algos:
    for env in envs:
        for seed in seeds:
            # edit file
            newFileLines = []
            outFile = open(fPath, 'w')
            for line in lines:
                if(line.find('obj')>=0):
                    line = f'obj: {algo}\n'
                if(line.find('env_name')>=0):
                    print('found env')
                    line = f'  env_name: {env}\n'
                if(line.find('seed:')>=0):
                    line = 'seed: {}\n'.format(seed)
                newFileLines.append(line)
            outFile.writelines(newFileLines)
            outFile.close()
            # launch experiment
            print('launching')
            x = multiprocessing.Process(target=run_experiment)
            x.start()
            time.sleep(2)
            process_list.append(x)

for x in process_list:
    x.join()


