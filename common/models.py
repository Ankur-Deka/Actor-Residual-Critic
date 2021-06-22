
# '''
# Code built on top of https://github.com/KamyarGh/rl_swiss 
# Refer[Original Code]: https://github.com/KamyarGh/rl_swiss
# '''
# rl_swiss/rlkit/torch/irl/disc_models/simple_disc_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation_class(hid_act):
    if hid_act == 'relu':
        hid_act_class = nn.ReLU
    elif hid_act == 'tanh':
        hid_act_class = nn.Tanh
    elif hid_act == 'leaky_relu':
        hid_act_class = nn.LeakyReLU
    else:
        raise NotImplementedError()
    return hid_act_class

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layer_blocks=2,
        hid_dim=100,
        hid_act='relu',
        use_bn=True,
        **kwargs
    ):
        super().__init__()
        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        elif hid_act == 'leaky_relu':
            hid_act_class = nn.LeakyReLU
        else:
            raise NotImplementedError()

        
        self.mod_list = nn.ModuleList([nn.Linear(input_dim, hid_dim)])
        if use_bn: self.mod_list.append(nn.BatchNorm1d(hid_dim))
        self.mod_list.append(hid_act_class())

        for i in range(num_layer_blocks - 1):
            self.mod_list.append(nn.Linear(hid_dim, hid_dim))
            if use_bn: self.mod_list.append(nn.BatchNorm1d(hid_dim))
            self.mod_list.append(hid_act_class())
        
        self.model = nn.Sequential(*self.mod_list)


    def forward(self, batch):
        output = self.model(batch)
        return output


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layer_blocks=2,
        hid_dim=100,
        hid_act='relu',
        use_bn=True,
        device=torch.device('cpu'),
        **kwargs
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        elif hid_act == 'leaky_relu':
            hid_act_class = nn.LeakyReLU
        else:
            raise NotImplementedError()

        self.input_dim = input_dim
        self.device = device

        self.first_fc = nn.Linear(input_dim, hid_dim)
        self.blocks_list = nn.ModuleList()

        for i in range(num_layer_blocks - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hid_dim, hid_dim))
            if use_bn: block.append(nn.BatchNorm1d(hid_dim))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        


    def forward(self, batch):
        x = self.first_fc(batch)
        for block in self.blocks_list:
            x = x + block(x)
        return x  
        # before sigmoid. i.e. output = log(D(s)) - log(1-D(s)) where D(s) is the original Discriminator
