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
from src.utils import get_dataset, average_weights, setup_seed, average_weights_test
from src.update import LocalUpdate, test_inference
import copy
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

CONFIG_PATH = './configs/'

def load_config(config_name):
    if config_name == 'default.yaml':
        config_name = CONFIG_PATH + 'default.yaml'
    else:
        config_name = config_name[0]
    with open(config_name) as f:
        config = yaml.safe_load(f)

    return config

if __name__ == "__main__":
    args = args_parser()
    print(args.config)
    config = load_config(args.config)
    print(config)
    device = 'cuda:'+ str(config['gpu']) if config['gpu'] is not None else 'cpu'
    print(device)
    if config['gpu']:
        torch.cuda.set_device(device)

    base_directory = './save/{}_{}_{}_{}_E[{}]_C[{}_{}]_iid[{}]_{}[{}]_LE[{}]_BS[{}]_lr[{}]_mu[{}]/'.\
        format(config['training_method'], config['optimizer'],config['model'],  'fedprox' if config['fedprox'] else '',config['epochs'], config['num_users'], config['frac'], config['iid'], 'sp' if config['alpha'] is None else 'alpha',config['shards_per_client'] if config['alpha'] is None else config['alpha'],
                config['local_ep'], config['local_bs'],config['lr'], config['mu'])

    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])])
    val_dataset = ImageFolder("/work/FAC/HEC/DESI/yshresth/aim/samgain/OCT_federated_classification/OCT2017 /val", transform=transform)

    for seed in config['seed']:
        print(f'Random seed: {seed}')
        setup_seed(seed)
        global_model = Network(model=config['model'], weights=config['weights'], train_backbone=False)
        train_dataset, test_dataset, user_groups, fed_averaging_weights = get_dataset(config, seed)
        print(len(train_dataset))
        global_model.to(device)
        global_model.train()

        for key, data in global_model.named_parameters():
            global_model.control[key] = torch.zeros_like(data.data)
            global_model.delta_control[key] = torch.zeros_like(data.data)
            global_model.delta_y[key] = torch.zeros_like(data.data)

        global_weights = global_model.state_dict()
        local_models = []
        local_weights = []# store local weights of all users for averaging
        local_states = []# store local states of all users, these parameters should not be uploaded
        local_updates = []
        
        for idx in range(config['num_users']):
            local_models.append(copy.deepcopy(global_model))
            local_states.append(copy.deepcopy(global_model.Get_Local_State_Dict()))
            local_weights.append(copy.deepcopy(global_weights))
            local_updates.append(LocalUpdate(config=config, dataset=train_dataset, idxs=user_groups[idx]))


        print('Starting training')

        min_val_loss = 500000
        best_model = copy.deepcopy(global_model)
        for epoch in tqdm(range(config['epochs'])):
            selected_clients = np.random.choice(range(config['num_users']), int(config['frac'] * config['num_users']), replace = False)
            print(selected_clients)
            for idx in selected_clients:
                # send global model to the client if we use feedavg, or if we want to train clients independently then we send previoudly trained client 
                local_model = copy.deepcopy(global_model)
                # LocalUpdate(config=config, dataset=train_dataset, idxs=user_groups[idx])
                w = local_updates[idx].update_weights(model=local_model, client_idx = idx, global_round = epoch)
                
                local_states[idx] = copy.deepcopy(local_model.Get_Local_State_Dict())
                local_models[idx] = copy.deepcopy(local_model)
                local_weights[idx] = copy.deepcopy(w)

                accuracy, loss, _ = test_inference(config, local_model, test_dataset)
                print(f"Global round {epoch}, Accuracy on test dataset for client {idx}: {accuracy}  {loss}")



            # average all the models
            selected_clients_models = [local_models[idx] for idx in selected_clients]
            selected_clients_weights = [local_weights[idx] for idx in selected_clients]
            dataset_weights = [fed_averaging_weights[idx] for idx in selected_clients]
            global_model = average_weights(selected_clients_models, dataset_weights, config)

            accuracy, loss, _ = test_inference(config, global_model, test_dataset)
            val_accuracy, val_loss, _ = test_inference(config, global_model, val_dataset)
            print(f"Global round {epoch}, Accuracy on test dataset: {accuracy}  {loss}, val dataset: {val_accuracy}, {val_loss}")

            if val_loss < min_val_loss:
                print('Best model found')
                min_val_loss = val_loss
                best_model = copy.deepcopy(global_model)
                
        torch.save({
            # 'client_models': local_models,
            'global_model': best_model, 
        }, f"{base_directory}model_{seed}.pt")

        with open(base_directory+f"config_{seed}.yaml", 'w') as f:
            yaml.dump(config, f)