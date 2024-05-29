from src.sampling import iid_split_dataset, noniid_split_dataset, Dirichlet_noniid
from numpy.random import RandomState
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import copy
import random
import numpy as np
import torch
import os

def get_dataset(config,seed=None):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    rs = RandomState(seed)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])])

    train_dataset = ImageFolder("/work/FAC/HEC/DESI/yshresth/aim/samgain/OCT_federated_classification/OCT2017 /train", transform=transform)
    test_dataset = ImageFolder("/work/FAC/HEC/DESI/yshresth/aim/samgain/OCT_federated_classification/OCT2017 /test", transform=transform)
    print(len(set([os.path.basename(x[0]) for x in train_dataset.imgs])))

    # sample training data amongst users
    if config['iid']:
        # Sample IID user data from Mnist
        user_groups = iid_split_dataset(train_dataset, config['num_users'], rs)
    else:
        # Sample Non-IID user data from Mnist
        if config['alpha'] is not None:
            user_groups,_ = Dirichlet_noniid(train_dataset, config['num_users'],config['alpha'],rs)
            # user_groups_test,_ = Dirichlet_noniid(test_dataset, config['num_users'],config['alpha'],rs)
        elif config['unequal']:
            # Chose uneuqal splits for every user
            raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups = noniid_split_dataset(train_dataset, config['num_users'],config['shards_per_client'],rs)
            # user_groups_test = noniid_split_dataset(test_dataset, config['num_users'],config['shards_per_client'],rs)
    
    config['num_users']=len(user_groups.keys())
    fed_averaging_weights = []

    for i in range(config['num_users']):
        fed_averaging_weights.append(len(user_groups[i])/len(train_dataset))
    
    
    return train_dataset, test_dataset, user_groups, np.array(fed_averaging_weights)



def average_weights(client_models, dataset_weights, config):
    global_model = copy.deepcopy(client_models[0])
    num_clients = len(client_models)

    assert len(client_models) == len(dataset_weights)

    model_parameters = []
    for client_model in client_models:
        model_parameters.append(list(client_model.parameters()))
    for param_idx, param in enumerate(global_model.parameters()):
        averaged_param = torch.zeros_like(param.data)
        for client_idx in range(num_clients):
            averaged_param += dataset_weights[client_idx] / np.sum(dataset_weights) * model_parameters[client_idx][param_idx]
        param.data = averaged_param

    if config['optimizer'] == 'scaffold':
        c = {} # control variable
        for client_model in client_models:
            for name, value in client_model.named_parameters():
                c[name] = torch.zeros_like(value.data)

            for name, value in client_model.named_parameters():
                c[name] += client_model.delta_control[name] / num_clients

        for name, _ in global_model.named_parameters():
            global_model.control[name].data = 1 / config['num_users'] * c[name]

    return global_model


def average_weights_test(w,omega=None):
    """
    Returns the average of the weights.
    """
    if omega is None:
        # default : all weights are equal
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg
        #omega = np.ones(len(w))
    omega = omega/np.sum(omega)
    w_avg = copy.deepcopy(w[0])
    for key in w[0].keys():
        avg_molecule = 0
        for i in range(len(w)):
            avg_molecule+=w[i][key]*omega[i]
        w_avg[key] = copy.deepcopy(avg_molecule)
    return w_avg


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True