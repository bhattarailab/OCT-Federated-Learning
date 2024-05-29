#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from collections import Counter
from torch.distributions.dirichlet import Dirichlet
import torch
import cvxopt
from cvxopt import matrix,solvers
import json
from tqdm import tqdm
import torch.utils.data as Data

def iid_split_dataset(dataset, num_users,rs):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)//num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(rs.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid_split_dataset(dataset, num_users,shards_per_client,rs):
    """
    Sample non-I.I.D client data from OCT dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards = shards_per_client*num_users
    num_imgs =  len(dataset)//num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([],dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    minimum_length = min(len(idxs), len(labels)) # just some hacky stuff
    idxs, labels = idxs[:minimum_length], labels[:minimum_length]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(rs.choice(idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        rs.shuffle(dict_users[i])
    return dict_users

def Dirichlet_noniid(dataset,num_users,alpha,rs):
    """
    Sample dataset with dirichlet distribution and concentration parameter alpha
    """
    # img_num_per_client = len(dataset)//num_users
    dict_users = {i: np.array([],dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    num_classes = np.max(labels)+1
    labels_idxs = []
    prior_class_distribution = np.zeros(num_classes)
    b = np.zeros(num_classes)
    for i in range(num_classes):
        labels_idxs.append(idxs[labels==i])
        prior_class_distribution[i] = len(labels_idxs[i])/len(dataset)
        b[i]=len(labels_idxs[i])
    
    data_ratio = np.zeros([num_classes,num_users])
    if isinstance(alpha,list):
        for i in range(num_users):
            data_ratio[:,i] = rs.dirichlet(prior_class_distribution*alpha[i])
    else:
        data_ratio = np.transpose(rs.dirichlet(prior_class_distribution*alpha,size=num_users))
    # data_ratio = data_ratio/np.sum(data_ratio,axis=1,keepdims=True)
    # Client_DataSize = len(dataset)//num_users*np.ones([num_users,1],dtype=np.int64)
    A = matrix(data_ratio)
    b = matrix(b)
    G = matrix(-np.eye(num_users))
    h = matrix(np.zeros([num_users,1]))
    P = matrix(np.eye(num_users))
    q = matrix(np.zeros([num_users,1]))
    results = solvers.qp(P,q,G,h,A,b)
    Client_DataSize = np.array(results['x'])
    # print(Client_DataSize)
    Data_Division = data_ratio*np.transpose(Client_DataSize)
    rest = []
    for label in range(num_classes):
        for client in range(num_users):
            data_idx = rs.choice(labels_idxs[label],int(Data_Division[label,client]),replace=False)
            dict_users[client] = np.concatenate([dict_users[client],data_idx],0)
            labels_idxs[label] = list(set(labels_idxs[label])-set(data_idx))
        rest = rest+labels_idxs[label]

    rest_clients = rs.choice(range(num_users),len(rest),replace = True)
    
    for n,user in enumerate(rest_clients):
        dict_users[user] = np.append(dict_users[user],rest[n])

    for user in range(num_users):
        rs.shuffle(dict_users[user])
    # print(data_ratio[:,:10])
    return dict_users,data_ratio
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]

    # prior_class_stat = Counter(labels)
    # # print(prior_class_stat)
    # prior_class_ditribution = np.zeros(max(labels)+1)
    # for label in range(max(labels)+1):
    #     prior_class_ditribution[label] = prior_class_stat[label]/len(dataset)
    # dist = Dirichlet(torch.tensor(alpha)*torch.tensor(prior_class_ditribution))
    # for i in range(num_users):
    #     class_distribution = dist.sample().detach().numpy()
    #     img_num_per_class = (img_num_per_client*class_distribution).astype(np.int64)
    #     img_num_per_class[-1]+=img_num_per_client-np.sum(img_num_per_class,dtype=np.int64)
    #     # print(img_num_per_class)
    #     label_start = 0
    #     for l in range(max(labels)+1):
    #         dict_users[i] =np.concatenate([dict_users[i],
    #                                       rs.choice(idxs[label_start:label_start+prior_class_stat[l]],
    #                                                        img_num_per_class[l],replace = False)],0)
    #         label_start = label_start+prior_class_stat[l]
    #     rs.shuffle(dict_users[i])
    # return dict_users
    

