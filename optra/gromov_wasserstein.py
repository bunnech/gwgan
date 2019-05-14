#!/usr/bin/python
# modified: https://github.com/rflamary/POT/blob/master/ot/gromov.py
# author: Charlotte Bunne <bunnec@ethz.ch>

# imports
import torch
from optra.sinkhorn_stab import sinkhorn_stabilized


def init_matrix(c1, c2, T, p, q, loss_fun='square_loss', cuda=False):
    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2) / 2

        def f2(b):
            return (b**2) / 2

        def h1(a):
            return a

        def h2(b):
            return b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * torch.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return torch.log(b + 1e-15)

    oq = torch.ones(len(q)).view(1, -1)
    op = torch.ones(len(p)).view(-1, 1)
    if cuda:
        oq, op = oq.cuda(), op.cuda()

    constc1 = torch.mm(torch.mm(f1(c1), p.view(-1, 1)), oq)
    constc2 = torch.mm(op, torch.mm(q.view(1, -1), torch.t(f2(c2))))
    constC = constc1 + constc2
    hc1 = h1(c1)
    hc2 = h2(c2)
    return constC, hc1, hc2


def tensor_product(constC, hC1, hC2, T):
    A = - torch.mm(hC1, T).mm(torch.t(hC2))
    tens = constC + A
    return tens


def gwloss(constC, hC1, hC2, T):
    tens = tensor_product(constC, hC1, hC2, T)
    return torch.sum(tens * T)


def gwggrad(constC, hC1, hC2, T):
    return 2 * tensor_product(constC, hC1, hC2, T)


def entropic_gromov_wasserstein(c1, c2, p, q, loss_fun, epsilon, niter=100,
                                tol=1e-9, verbose=False, coupling=True,
                                cuda=False):
    c1, c1_norm = c1
    c2, c2_norm = c2

    if cuda:
        p, q = p.cuda(), q.cuda()

    # initialisation
    T = torch.ger(p, q)
    constC, hc1, hc2 = init_matrix(c1_norm, c2_norm, T, p, q, loss_fun, cuda=cuda)

    iter = 0
    err = 1

    while (err > tol and iter < niter):
        Tprev = T

        # compute the gradient
        tens = gwggrad(constC.detach(), hc1.detach(), hc2.detach(), T)

        T = sinkhorn_stabilized(p, q, tens, epsilon,
                                method='sinkhorn_stabilized', numItermax=niter,
                                cuda=cuda)

        if iter % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = torch.norm(T - Tprev)

            if verbose:
                if iter % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(iter, err))

        iter += 1

    # recomputaiton of init matrix, T as constant
    constC, hc1, hc2 = init_matrix(c1, c2, T, p, q, loss_fun, cuda=cuda)

    # gromov-wasserstein transport: T
    # computation of romov-wasserstein discrepancy
    gw_loss = gwloss(constC, hc1, hc2, T)

    if coupling:
        return gw_loss, T
    else:
        return gw_loss
