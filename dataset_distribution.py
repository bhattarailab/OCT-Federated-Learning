"""
TODO:
    - Configuration file 
    - Dataset split 
"""


import torch
from tqdm import tqdm
import numpy as np
from src.options import args_parser
from src.models import Network
import yaml
from src.utils import get_dataset, average_weights, setup_seed
from src.update import LocalUpdate, test_inference
import copy
import os
from collections import defaultdict
import re
import functools
import matplotlib.pyplot as plt
CONFIG_PATH = './configs/'

def load_config(config_name):
    with open(CONFIG_PATH + config_name) as f:
        config = yaml.safe_load(f)

    return config

if __name__ == "__main__":
    args = args_parser()
    config = load_config(args.config)
    print(config)
    device = 'cuda:'+ str(config['gpu']) if config['gpu'] is not None else 'cpu'
    print(device)
    if config['gpu']:
        torch.cuda.set_device(device)

    for seed in config['seed']:
        print(f'Random seed: {seed}')
        setup_seed(seed)
        # global_model = Network(model=config['model'], weights=config['weights'], train_backbone=config['pretrained'])
        train_dataset, test_dataset, user_groups, fed_averaging_weights = get_dataset(config, seed=seed)
        # print(user_groups)

        clients = []
        count = [0] * 10
        for _ in range(config['num_users']):
            clients.append(defaultdict(int))
        print(user_groups.keys())
        for key in user_groups.keys():
            for index in user_groups[key]:
                path = os.path.basename(train_dataset.imgs[index][0])
                id = int(path.split('-')[1])
                clients[key][id] += 1
                count[key] += 1

        keys = list(map(lambda x: set(x.keys()), clients))
        
        for i in range(len(keys)):
            for j in range(len(keys)):
                if i == j: 
                    continue
                else:
                    intersection = keys[i] & keys[j]
                    if len(intersection) != 0:
                        print(i, j, intersection, [(clients[i][x], clients[j][x]) for x in intersection])

        print(count, sum(count), len(train_dataset))
        
        print('Dataset distribution')
        label_counts = []
        for _ in range(config['num_users']):
            label_counts.append(defaultdict(list, {k: 0 for k in range(4)}))

        for key in user_groups.keys():
            for index in tqdm(user_groups[key]):
                label_counts[key][train_dataset[index][1]] += 1

            print(label_counts[key])


        # clients = [0, 1, 4, 6]
        # classes = train_dataset.classes
        # nrows = 1
        # ncols = 4

        # fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True)
        # fig.set_figwidth(ncols * 3)
        # fig.set_figheight(nrows * 4)
        # itr = 0
        # X_axis = np.arange(len(label_counts[0].keys()))
        # for idx, client in enumerate(clients):
        #     axs[itr].bar(label_counts[clients].keys(), label_counts[client].values(), 0.4, label='2 SPC')
        #     axs[itr].tick_params(axis='x', which='major', labelsize=10, rotation=0)
        #     itr += 1

        # plt.legend()
        # fig.tight_layout()
        # plt.savefig('distribution.png', dpi=300)
        # plt.show()
        
        # break