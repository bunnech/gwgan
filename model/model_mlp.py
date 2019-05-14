#!/usr/bin/python

# imports
import torch


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.map1 = torch.nn.Linear(256, 128, bias=True)
        self.map2 = torch.nn.Linear(128, 128, bias=True)
        self.map3 = torch.nn.Linear(128, 128, bias=True)
        self.map4 = torch.nn.Linear(128, 2, bias=True)

    def forward(self, x):
        x = torch.nn.functional.relu(self.map1(x))
        x = torch.nn.functional.relu(self.map2(x))
        x = torch.nn.functional.relu(self.map3(x))
        return self.map4(x)


class Adversary(torch.nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()
        self.map0 = torch.nn.Linear(3, 32, bias=True)
        self.map1 = torch.nn.Linear(2, 32, bias=True)
        self.map2 = torch.nn.Linear(32, 16, bias=True)
        self.map3 = torch.nn.Linear(16, 8, bias=True)
        self.map4 = torch.nn.Linear(8, 2, bias=True)
        self.map5 = torch.nn.Linear(8, 3, bias=True)

    def forward(self, x, data=False):
        if data:
            x = torch.nn.functional.relu(self.map0(4*x))
        else:
            x = torch.nn.functional.relu(self.map1(4*x))
        x = torch.nn.functional.relu(self.map2(x))
        x = torch.nn.functional.relu(self.map3(x))
        if data:
            return self.map5(x)
        else:
            return self.map4(x)


def weights_init_adversary(m):
    # custom weights initialization called on generator and discriminator
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_generator(m):
    # custom weights initialization called on generator and discriminator
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
