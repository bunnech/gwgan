#!/usr/bin/python
# author: Charlotte Bunne <bunnec@ethz.ch>

# imports
import torch
from optra.gromov_wasserstein import entropic_gromov_wasserstein


def gwnorm_distance(D_x, D_g, epsilon, niter, loss_fun='square_loss',
                    coupling=True, cuda=False):
    p = torch.ones((D_x[0].shape[1], ))
    p /= torch.numel(p)
    q = torch.ones((D_g[0].shape[1], ))
    q /= torch.numel(q)

    gw_x_x, _ = entropic_gromov_wasserstein(D_x, D_x, p, p,
                                         loss_fun, epsilon, niter,
                                         coupling=coupling, cuda=cuda)
    gw_x_g, T = entropic_gromov_wasserstein(D_x, D_g, p, q,
                                         loss_fun, epsilon, niter,
                                         coupling=coupling, cuda=cuda)
    gw_g_g, _ = entropic_gromov_wasserstein(D_g, D_g, q, q,
                                         loss_fun, epsilon, niter,
                                         coupling=coupling, cuda=cuda)

    # compute normalized Gromow-Wasserstein distance
    return 2*gw_x_g - gw_x_x - gw_g_g, T


def loss_total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


def loss_procrustes(f_x, x, cuda):
    m = torch.mm(torch.t(f_x.detach()), x)
    u, _, v = torch.svd(m.cpu())
    p = torch.mm(u, torch.t(v))
    if cuda:
        return torch.norm(f_x - torch.mm(x, torch.t(p.cuda()))) ** 2
    else:
        return torch.norm(f_x - torch.mm(x, torch.t(p))) ** 2
