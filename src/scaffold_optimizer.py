# -*- coding:utf-8 -*-
"""
@Time: 2022/03/02 13:34
@Author: KI
@File: ScaffoldOptimizer.py
@Motto: Hungry And Humble
"""
from torch.optim import Optimizer


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for param, contron_variable_server, contron_variable_i in zip(group['params'], server_controls.values(), client_controls.values()):
                if param.grad is None:
                    continue
                dp = param.grad.data + contron_variable_server.data - contron_variable_i.data
                param.data = param.data - dp.data * group['lr']

        return loss