#!/usr/bin/python
# author: Charlotte Bunne <bunnec@ethz.ch>

# imports
import os
import torch
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model.data import *

FUNCTION_MAP = {'4mode': gaussians_4mode,
                '5mode': gaussians_5mode,
                '8mode': gaussians_8mode,
                '3d_4mode': gaussians_3d_4mode
                }

def get_args():
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_iter', type=int, default=10000)
    parser.add_argument('--id', type=str)

    # arguments to choose dataset (mnist, fmnist, cifar-gray etc.)
    parser.add_argument('--data', default='mnist')
    parser.add_argument('--beta', type=float, default=40)
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')

    # arguments for Gaussian mixture application
    parser.add_argument('--modes', type=str, choices=FUNCTION_MAP.keys())
    parser.add_argument('--l1reg', action='store_true')
    parser.add_argument('--advsy', action='store_true')

    return parser.parse_args()


def sample_z(m, n):
    x = torch.Tensor(m, n)
    return x.normal_(mean=0, std=1)


def reset_grad(net1, net2):
    net1.zero_grad()
    net2.zero_grad()


def cdist(u, metric='euclidean'):
    """
    Computes distance between each pair of the two collections of inputs.
    """
    if metric == 'euclidean':
        return euclidean_distance(u, u)
    elif metric == 'sqeuclidean':
        return euclidean_distance(u, u, squared=True)
    else:
        raise ValueError('metric not implemented yet')


def euclidean_distance(x, y, squared=False):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)

    # replace NaNs by 0
    dist[torch.isnan(dist)] = 0
    # add small value to avoid numerical issues
    dist = dist + 1e-16
    if squared:
        return dist ** 2
    else:
        return dist


def get_inner_distances(s, metric='euclidean', concat=True):
    if concat is True:
        s1, s2 = torch.chunk(s, 2, dim=0)
        return cdist(s1, metric), cdist(s2, metric)
    else:
        return cdist(s, metric)


def normalise_matrices(m):
    def normalisation(x):
        return x / torch.max(x)

    if len(m) == 2:
        # recover matrices
        m1, m2 = m

        if (torch.sum(torch.isnan(m1)) + torch.sum(torch.isnan(m2))) > 0:
            print('distance computation returns NaNs.')
        return normalisation(m1), normalisation(m2)
    else:
        if torch.sum(torch.isnan(m)) > 0:
            print('distance computation returns NaNs.')

        return normalisation(m)


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    return fig.colorbar(mappable, cax=cax)
