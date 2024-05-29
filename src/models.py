import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Network(nn.Module):
    def __init__(self, model='resnet18', weights = 'IMAGENET1K_V1', train_backbone=True):
        super(Network, self).__init__()
        if model == 'resnet18':
            self.backbone = models.resnet18(weights='ResNet18_Weights.' + weights)
        elif model == 'resnet50':
            self.backbone = models.resnet50(weights='ResNet50_Weights.' + weights)

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4, bias=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = train_backbone
            
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        for param in self.backbone.layer4[1].parameters():
            param.requires_grad = True

        # for scaffold
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}



    def forward(self, x):
        return self.backbone(x)


    def Get_Local_State_Dict(self):
        # save local parameters without weights and bias
        sd = self.state_dict()
        for name in list(sd.keys()):
            if 'weight' in name or 'bias' in name:
                sd.pop(name)
        return sd

    def Load_Local_State_Dict(self,local_dict):
        # load local parameters saved by Get_Local_State_Dict()
        sd = self.state_dict()
        sd.update(local_dict)
        self.load_state_dict(sd)
