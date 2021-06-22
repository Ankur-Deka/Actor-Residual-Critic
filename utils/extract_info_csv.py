import os
import sys
import numpy as np
import pandas as pd
import yaml
import copy
import argparse

def recurse_func(param, l, info_dict):
    if info_dict[param] in (True, False):
        if info_dict[param]==True:
            ans = copy.copy(l)
            ans.append(param)
            return [ans]
        else:
            return []
        
    else:
        ans = []
        l.append(param)
        for k,v in info_dict[param].items():
            ans += recurse_func(k,l,info_dict[param])
        l = l[:-1]
        return ans

# returns list of list of keys
def get_key_list(info_dict):
    ans = []
    for k,v in info_dict.items():
        ans += recurse_func(k,[],info_dict)
    return ans

# mereges keys in inner list
def get_key_names(key_list):
    ans = []
    for l in key_list:
        n = l[0]
        for i in range(1,len(l)):
            n += '_' + l[i]
        ans.append(n)
    return ans

# load only those configs that are present in key_list
def load_config(key_list, config_dict):
    ans = {}
    for l in key_list:
        d = config_dict[l[0]] if l[0] in config_dict else config_dict
        n = l[0]
        
        for i in range(1,len(l)):
            n += '_' + l[i]
            d = d[l[i]] if l[i] in d else d

        ans[n] = None if type(d) is dict else d
    return ans

def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def extract_all_data(root_dir, info_file = None):
    if info_file is None:
        info_file = os.path.join(os.path.dirname(__file__), 'extract_info.yml')
    runs_list = os.listdir(root_dir)

    # info_dict
    info_dict = yaml.load(open(info_file, 'r'))
    key_list = get_key_list(info_dict)

    # initialize final dict

    progress_keys = ['Running Env Steps', 'Real Sto Return', 'Real Det Return']
    headings = ['Name'] + get_key_names(key_list) + progress_keys 
    final_dict = {k:[] for k in headings}

    for run_dir in runs_list:
        if os.path.isdir(os.path.join(root_dir, run_dir)):
            run_path = os.path.join(root_dir, run_dir)
            run_dict = {}

            # reward, timesteps
            progress_file = os.path.join(run_path, 'progress.csv')
            if is_non_zero_file(progress_file):
                run_dict['Name'] = run_dir

                progress = pd.read_csv(progress_file)
                for k in progress_keys:
                    if k in progress:
                        run_dict[k] = progress.iloc[-1][k]
                    else:
                        run_dict[k] = None
                # config dict
                files = os.listdir(run_path)
                config_file = [os.path.join(run_path,f) for f in files if f.endswith('.yml')][0]
                config_dict = yaml.load(open(config_file,'r')) 
                run_dict.update(load_config(key_list, config_dict))

                # add to final dict
                for k,v in run_dict.items():
                    final_dict[k].append(v)

    return pd.DataFrame(final_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, help='Root run directory of experiments')
    parser.add_argument('--info_file', default=None, help='Which info to extract')
    parser.add_argument('--out_file', default=None, help='Where to save all the data')
    args = parser.parse_args()
    if args.out_file is None:
        args.out_file = os.path.join(args.root_dir, 'info.csv')

    df = extract_all_data(args.root_dir, args.info_file)
    df.to_csv(args.out_file, index=False)
    print(df)
