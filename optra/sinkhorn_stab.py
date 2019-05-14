#!/usr/bin/python
# modified: https://github.com/rflamary/POT/blob/master/ot/bregman.py
# author: Charlotte Bunne <bunnec@ethz.ch>


# imports
import torch


def sinkhorn_stabilized(a, b, M, reg, numItermax=1000, tau=1e3, stopThr=1e-9,
                        warmstart=None, verbose=False, print_period=20,
                        log=False, cuda=False, **kwargs):

    if len(a) == 0:
        a = torch.ones((M.shape[0],)) / M.shape[0]
    if len(b) == 0:
        b = torch.ones((M.shape[1],)) / M.shape[1]

    # test if multiple target
    if len(b.shape) > 1:
        nbb = b.shape[1]
        a = a.unsqueeze(1)
    else:
        nbb = 0

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of distances
    if warmstart is None:
        alpha, beta = torch.zeros(na), torch.zeros(nb)
    else:
        alpha, beta = warmstart

    if nbb:
        u = torch.ones(na, nbb) / na
        v = torch.ones(nb, nbb) / nb
    else:
        u = torch.ones(na) / na
        v = torch.ones(nb) / nb

    if cuda:
        u, v = u.cuda(), v.cuda()
        alpha, beta = alpha.cuda(), beta.cuda()

    def get_K(alpha, beta):
        """log space computation"""
        return torch.exp(-(M - alpha.reshape(na, 1) -
                        beta.reshape(1, nb)) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return torch.exp(-(M - alpha.reshape(na, 1) - beta.reshape(1, nb)) /
                      reg + torch.log(u.reshape(na, 1)) + torch.log(v.reshape(1, nb)))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1

    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (torch.mv(torch.t(K), u) + 1e-16)
        u = a / (torch.mv(K, v) + 1e-16)

        # remove numerical problems and store them in K
        if torch.abs(u).max() > tau or torch.abs(v).max() > tau:
            if nbb:
                alpha, beta = alpha + reg * \
                    torch.max(torch.log(u), 1), beta + reg * torch.max(torch.log(v))
            else:
                alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)
                if nbb:
                    u, v = torch.ones((na, nbb)) / na, torch.ones((nb, nbb)) / nb
                else:
                    u, v = torch.ones(na) / na, torch.ones(nb) / nb
                if cuda:
                    u, v = u.cuda(), v.cuda()
            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = torch.sum((u - uprev)**2) / torch.sum((u)**2) + \
                    torch.sum((v - vprev)**2) / torch.sum((v)**2)
            else:
                transp = get_Gamma(alpha, beta, u, v)
                err = torch.norm((torch.sum(transp, dim=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 20) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if torch.sum((u != u) == 1) > 0 or torch.sum((v != v) == 1) > 0:
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1

    # print('err=',err,' cpt=',cpt)
    if log:
        log['logu'] = alpha / reg + torch.log(u)
        log['logv'] = beta / reg + torch.log(v)
        log['alpha'] = alpha + reg * torch.log(u)
        log['beta'] = beta + reg * torch.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])
        if nbb:
            res = torch.zeros((nbb))
            for i in range(nbb):
                res[i] = torch.sum(get_Gamma(alpha, beta, u[:, i], v[:, i])
                                   * M)
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if nbb:
            res = torch.zeros((nbb))
            for i in range(nbb):
                res[i] = torch.sum(get_Gamma(alpha, beta, u[:, i], v[:, i])
                                   * M)
            return res
        else:
            return get_Gamma(alpha, beta, u, v)
