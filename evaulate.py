import torch
from tqdm import tqdm
import numpy as np
from src.models import Network
import yaml
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from src.update import test_inference
from src.utils import setup_seed
import pathlib
import torchvision.transforms as transforms

def load_config(config_name):
    with open(config_name) as f:
        config = yaml.safe_load(f)

    return config

if __name__ == "__main__":
    test_path = '/kaggle/input/kermany2018/OCT2017 /test'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])])
    test_dataset = ImageFolder("/work/FAC/HEC/DESI/yshresth/aim/samgain/OCT_federated_classification/OCT2017 /test", transform=transform)
    print(len(test_dataset))

    base_directory = './save/fedavg_sgd_resnet18__E[100]_C[10_0.5]_iid[None]_sp[2]_LE[2]_BS[64]_lr[0.01]_mu[0.0]'

    accuracies = []
    f1_scores = []
    seeds = [1, 2, 3]
    

    config = load_config(base_directory + f'/config_{seeds[0]}.yaml')
    for seed in seeds:
        config['seed'] = seed
        setup_seed(seed=seed)
        saved_objects = torch.load(f'{base_directory}/model_{seed}.pt')
        accuracy, _, f1 = test_inference(config, saved_objects['global_model'], test_dataset)
        
        accuracies.append(accuracy * 100)
        f1_scores.append(f1)

        print(f'Global model accuracy: {accuracy * 100:.3f}, F1 score: {f1:.3f}')

        # models = []
        # print(f"For seed {seed}:")
        # for idx, model in enumerate(saved_objects['client_models']):
        #     models.append(model)
        #     accuracy, _ = test_inference(config, model, test_dataset)
        #     print(f"Client {idx}, Accuracy: {accuracy}")

    print(round(np.mean(accuracies), 3), round(np.std(accuracies), 3))
    print(round(np.mean(f1_scores), 3), round(np.std(f1_scores), 3))
