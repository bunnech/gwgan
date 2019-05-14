#!/usr/bin/python
# network architectures from infoGAN (https://arxiv.org/abs/1606.03657)

# imports
import torch.nn as nn
import torch
torch.set_default_dtype(torch.double)


class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size//4) * (self.input_size//4)),
            nn.BatchNorm1d(128 * (self.input_size//4) * (self.input_size//4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)
        return x


class Adversary(nn.Module):
    def __init__(self, input_dim=1, output_dim=1024, input_size=32):
        super(Adversary, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)
        return x


def weights_init_generator(net):
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)


def weights_init_adversary(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
