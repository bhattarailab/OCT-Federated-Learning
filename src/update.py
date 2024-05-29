#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm
from src.scaffold_optimizer import ScaffoldOptimizer
from sklearn.metrics import f1_score


def ce_criterion(pred, target, *args):
    ce_loss = F.cross_entropy(pred, target)
    return ce_loss, float(ce_loss)

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, config, dataset, idxs, global_round = 0,verbose = None):
        self.config = config
        # self.logger = logger
        self.trainloader, self.valloader, self.val_dataset = self.train_val(
            dataset, list(idxs))
        self.device = 'cuda:'+ str(config['gpu']) if config['gpu'] is not None else 'cpu'
        self.criterion = ce_criterion
        self.test_criterion = ce_criterion

    def train_val(self, dataset, idxs):
        """
        Returns train and test dataloaders for a given dataset and user indexes.
        """
        # split indexes for train, and test (80, 20)
        idxs_train = idxs[:int(0.85*len(idxs))]
        idxs_test = idxs[int(0.15*len(idxs)):]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.config['local_bs'], shuffle=True, num_workers=8)
        valloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max([int(len(idxs_test)/10),10]), shuffle=False, num_workers=8)
        return trainloader, valloader, DatasetSplit(dataset, idxs_test)

    def update_weights(self, model, client_idx, global_round=0, server_control = None):
        # Update model's weights or gates
        # Set mode to train model
        self.net = model.to(self.device)
        self.init_net = copy.deepcopy(self.net)
        # init_accuracy,init_test_loss = self.inference(self.net)
        self.net.train()
        np = self.net.parameters()
        # Set optimizer for the local updates

        if self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(np, lr=self.config['lr'],
                                        momentum=self.config['momentum'],weight_decay=self.config['reg'])
        elif self.config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(np, lr=self.config['lr'],
                                        weight_decay=self.config['reg'])
        elif self.config['optimizer'] == 'scaffold':
            print('Using scaffold optimizer')
            optimizer = ScaffoldOptimizer(np, lr=self.config['lr'], weight_decay=self.config['reg'])
        
        min_val_loss = 5000
        min_epochs = self.config['local_ep'] // 5
        best_model = None
        for iter in range(self.config['local_ep']):
            for batch_idx, (datas, labels) in enumerate(self.trainloader):
                
                datas, labels = datas.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                output = self.net(datas)
                total_loss,celoss = self.criterion(output, labels)
                if self.config['fedprox']:
                    PLoss = self.Proxy_Loss()
                    total_loss += 0.5*self.config['mu']*PLoss
                total_loss.backward()

                if self.config['optimizer'] == 'scaffold':
                    optimizer.step(self.init_net.control, self.net.control)
                else:
                    optimizer.step()
            # val_accuracy, val_loss = self.inference(self.net)
            # if best_model is None:
            #     best_model = copy.deepcopy(self.net)
            # if iter >= min_epochs and val_loss < min_val_loss:
            #     min_val_loss = val_loss
                # best_model = copy.deepcopy(self.net)
                # self.logger.add_scalar('weight_loss', total_loss.item())
        

        print('Client: {} | Global Round : {} | Local Epoch : {}| Iteration: {}\{} |\tLoss: {:.4f}'.format(client_idx, global_round, iter, batch_idx, len(self.trainloader), total_loss.item()))
        if self.config['optimizer'] == 'scaffold':
            ann = copy.deepcopy(self.net)
            # update c
            # c+ <- ci - c + 1/(steps * lr) * (x-yi)
            # save ann
            temp = {}
            for k, v in ann.named_parameters():
                if not v.requires_grad:
                    continue
                temp[k] = v.data.clone()

            for k, v in self.init_net.named_parameters():
                if not v.requires_grad:
                    continue
                local_steps = len(self.trainloader)
                ann.control[k] = ann.control[k] - self.init_net.control[k] + (v.data - temp[k]) / (local_steps * self.config['lr'])
                ann.delta_y[k] = temp[k] - v.data
                ann.delta_control[k] = ann.control[k] - self.init_net.control[k]

            self.net = ann
        return self.net.state_dict()

    @torch.no_grad
    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.to(self.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (datas, labels) in enumerate(self.valloader):
            datas, labels = datas.to(self.device), labels.to(self.device)
            # print(labels)
            # input()
            # Inference
            outputs = model(datas)
            batch_loss,_ = self.test_criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        model.train()
        accuracy = correct/total
        return accuracy, loss/(batch_idx+1)

    def Proxy_Loss(self):
        loss = 0.
        init_state_dict = self.init_net.state_dict()
        for name,p in self.net.named_parameters():
            if 'weight' in name or 'bias' in name:
                loss += torch.sum((p-init_state_dict[name])**2)
        return loss


@torch.no_grad
def test_inference(config, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda:'+ str(config['gpu']) if config['gpu'] is not None else 'cpu'
    model.to(device)
    criterion = F.cross_entropy
    testloader = DataLoader(test_dataset, batch_size=256,
                            shuffle=False)


    all_labels = np.array([])
    all_predicted = np.array([])
    for batch_idx, (datas, labels) in enumerate(testloader):
        datas, labels = datas.to(device), labels.to(device)

        # Inference
        outputs = model(datas)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        all_labels = np.hstack([all_labels.copy(), labels.cpu().numpy()])
        all_predicted = np.hstack([all_predicted.copy(), pred_labels.cpu().numpy()])

    accuracy = correct/total
    f1 = f1_score(all_labels, all_predicted, average='macro')
    return accuracy, loss/(batch_idx+1), f1