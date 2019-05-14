#!/usr/bin/python
# author: Charlotte Bunne

# imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import pandas as pd
import seaborn as sns
from pylab import array

# internal imports
from model.utils import *
from model.data import *
from model.model_mlp import Generator, Adversary
from model.model_mlp import weights_init_adversary, weights_init_generator
from model.loss import gwnorm_distance
from model.loss import loss_procrustes

# get arguments
FUNCTION_MAP = {'4mode': gaussians_4mode,
                '5mode': gaussians_5mode,
                '8mode': gaussians_8mode,
                '3d_4mode': gaussians_3d_4mode
                }
args = get_args()

# plotting preferences
matplotlib.rcParams['font.sans-serif'] = 'Times New Roman'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 10

# system preferences
torch.set_default_dtype(torch.double)
seed = np.random.randint(100)
np.random.seed(seed)
torch.manual_seed(seed)

# settings
batch_size = 256
z_dim = 256
lr = 0.0002
plot_every = 100
niter = 10
epsilon = 0.01
ngen = 10
if args.advsy:
    lam = {'4mode': 0.001, '5mode': 0.0001}
else:
    lam = 0.01
beta = 1
stop_adversary = args.num_iter
l1_reg = args.l1reg
learn_c = args.advsy
train_iter = args.num_iter
modes = args.modes

if l1_reg:
    model = 'gwgan_gaussian_l1_{}_adversary_{}_lam_{}_id_{}'\
     .format(modes, learn_c, lam, args.id)
else:
    model = 'gwgan_gaussian_{}_adversary_{}_id_{}'\
     .format(modes, learn_c, args.id)

simulation = FUNCTION_MAP[modes]

# data simulation
data, y = simulation(40000)
data_size = len(data)
data = np.concatenate((data, data[:batch_size, :]), axis=0)
y = np.concatenate((y, y[:batch_size]), axis=0)

save_fig_path = 'out_' + model
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)

real = data[:1000]
real_y = y[:1000]

fig1 = plt.figure(figsize=(2.4, 2))
if modes == '3d_4mode':
    df = pd.DataFrame({'x1': real[:, 0],
                       'x2': real[:, 1],
                       'x3': real[:, 2],
                       'in': real_y})
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter3D(df.x1, df.x2, df.x3, c='#1B263B')
    ax1.set_zlim([-4, 4])
    view_1 = (25, -135)
    view_2 = (25, -45)
    init_view = view_2
    ax1.view_init(*init_view)
    ax1.set_zlabel('x3')
else:
    df = pd.DataFrame({'x1': real[:, 0],
                       'x2': real[:, 1],
                       'in': real_y})
    ax1 = fig1.add_subplot(111)
    sns.kdeplot(df.x1, df.x2, shade=True, cmap='Blues', n_levels=20, legend=False)
ax1.set_xlim([-4, 4])
ax1.set_ylim([-4, 4])
ax1.set_title(r'target')
fig1.tight_layout()
fig1.savefig(save_fig_path + '/real.pdf')

# define networks and parameters
generator = Generator()
adversary = Adversary()

# weight initialisation
generator.apply(weights_init_generator)
adversary.apply(weights_init_adversary)

# create optimiser
g_optimizer = torch.optim.Adam(generator.parameters(), 5*lr)
c_optimizer = torch.optim.Adam(adversary.parameters(), lr)

# zero gradients
reset_grad(generator, adversary)

# sample for plotting
z_ex = sample_z(1000, z_dim)

# set iterator for plot numbering
i = 0

# learn with and without adversary
if learn_c:
    only_g = 0
else:
    only_g = train_iter

loss_history = list()
loss_orth = list()
loss_og = 0

for it in range(train_iter):
    train_c = ((it + 1) % (ngen + 1) == 0)

    start_idx = it * batch_size % data_size
    X_mb = data[start_idx:start_idx + batch_size, :]
    y_mb = y[start_idx:start_idx + batch_size]

    # sample two random numbers z from Z
    z = sample_z(batch_size, z_dim)

    # sample two data points from real data mini batch
    x = torch.from_numpy(X_mb[:batch_size, :])
    y_s = y_mb[:batch_size]

    if it <= only_g:
        for q in generator.parameters():
            q.requires_grad = True
        for p in adversary.parameters():
            p.requires_grad = False

        g = generator.forward(z)
        f_g = g
        f_x = x
    else:
        if train_c and it < stop_adversary:
            for q in generator.parameters():
                q.requires_grad = False
            for p in adversary.parameters():
                p.requires_grad = True

        else:
            for q in generator.parameters():
                q.requires_grad = True
            for p in adversary.parameters():
                p.requires_grad = False

        # result generator
        g = generator.forward(z)

        # result adversary
        f_x = adversary.forward(x)
        f_g = adversary.forward(g)

    # compute inner distances
    D_g = get_inner_distances(f_g, metric='euclidean', concat=False)
    D_x = get_inner_distances(f_x, metric='euclidean', concat=False)

    # distance matrix normalisation
    D_x_norm = normalise_matrices(D_x)
    D_g_norm = normalise_matrices(D_g)

    # compute normalized gromov-wasserstein distance
    loss_gw, T = gwnorm_distance((D_x, D_x_norm), (D_g, D_g_norm),
                                 epsilon, niter, loss_fun='square_loss',
                                 coupling=True)

    if it < only_g:
        # train generator
        if l1_reg:
            loss_gen = loss_gw + lam * (torch.norm(g, p=1) - 2)
        else:
            loss_gen = loss_gw
        loss_gen.backward()

        # parameter updates
        g_optimizer.step()

        # zero gradients
        g_optimizer.zero_grad()

    else:
        if train_c and it < stop_adversary:
            loss_og = loss_procrustes(f_x, x, cuda=False)
            loss_adv = -loss_gw + beta * loss_og
            # train adversary
            loss_adv.backward()

            # parameter updates
            c_optimizer.step()
            # zero gradients
            reset_grad(generator, adversary)

        else:
            # train generator
            if l1_reg:
                loss_gen = loss_gw + lam[modes] * (torch.norm(g, p=1) - 2)
            else:
                loss_gen = loss_gw
            loss_gen.backward()

            # parameter updates
            g_optimizer.step()
            # zero gradients
            reset_grad(generator, adversary)

    # plotting
    if (it+1) % plot_every == 0:
        # get generator example
        g_ex = generator.forward(z_ex)
        if it >= only_g:
            f_gx = adversary.forward(g_ex)
            f_gx = f_gx.detach().numpy()
            f_dx = adversary.forward(torch.from_numpy(real))
            f_dx = f_dx.detach().numpy()
        g_ex = g_ex.detach().numpy()

        # plotting
        fig2 = plt.figure(figsize=(2.4, 2))
        ax2 = fig2.add_subplot(111)
        result = pd.DataFrame({'x1': g_ex[:, 0],
                               'x2': g_ex[:, 1]})
        sns.kdeplot(result.x1, result.x2,
                    shade=True, cmap='Blues', n_levels=20, legend=False)
        # ax2.set_title(r'$g_\theta(Z)$')
        ax2.set_title(r'iteration {}'.format((it+1)))
        plt.tight_layout()
        fig2.savefig(os.path.join(save_fig_path, 'gen_{}.pdf'.format(
                     str(i).zfill(3))))

        if it >= only_g:
            fig6 = plt.figure(figsize=(4.5, 2))
            features = pd.DataFrame({'g1': f_gx[:, 0],
                                     'g2': f_gx[:, 1],
                                     'd1': f_dx[:, 0],
                                     'd2': f_dx[:, 1]
                                     })
            ax1 = fig6.add_subplot(121)
            sns.kdeplot(features.g1, features.g2,
                        shade=True, cmap='Greys', n_levels=20, legend=False)
            # ax1.set_title(r'$f_\omega(g_\theta(Z))$')
            ax1.set_xlim([-4, 4])
            ax1.set_ylim([-4, 4])
            ax1.set_title(r' ')

            ax2 = fig6.add_subplot(122)
            sns.kdeplot(features.d1, features.d2,
                        shade=True, cmap='Greys', n_levels=20, legend=False)
            ax2.set_xlim([-4, 4])
            ax2.set_ylim([-4, 4])
            ax2.set_title(r' ')
            plt.tight_layout(pad=1)
            fig6.savefig(os.path.join(save_fig_path, 'feature_{}.pdf'.format(
                         str(i).zfill(3))))

        fig3, ax = plt.subplots(1, 3, figsize=(6.9, 2))
        ax0 = ax[0].imshow(T.detach().numpy(), cmap='RdBu_r')
        colorbar(ax0)
        ax1 = ax[1].imshow(D_g.detach().numpy(), cmap='Blues')
        colorbar(ax1)
        ax2 = ax[2].imshow(D_x.detach().numpy(), cmap='Blues')
        colorbar(ax2)
        ax[0].set_title(r'$T$')
        ax[1].set_title(r'Pairwise Distances of $f_\omega(g_\theta(Z))$')
        ax[2].set_title(r'Pairwise Distances of $f_\omega(X)$')
        plt.tight_layout(pad=1)
        fig3.savefig(os.path.join(save_fig_path, 'ccc_{}.pdf'.format(
                str(i).zfill(3))), bbox_inches='tight')

        plt.close('all')
        print('iter:', it+1, 'GW loss:', loss_gw, 'Orth. loss', loss_og)
        i += 1

        loss_history.append(loss_gw)
        loss_orth.append(loss_og)


# plot loss history
fig4 = plt.figure(figsize=(2.4, 2))
ax4 = fig4.add_subplot(111)
ax4.plot(np.arange(len(loss_history))*100, loss_history, 'k.')
ax4.set_xlabel('Iterations')
ax4.set_ylabel(r'$\overline{GW}_\epsilon$ loss')
plt.tight_layout()
fig4.savefig(save_fig_path + '/loss_history.pdf')

fig5 = plt.figure(figsize=(2.4, 2))
ax5 = fig5.add_subplot(111)
ax5.plot(np.arange(len(loss_orth))*100, loss_orth, 'k.')
ax5.set_xlabel('Iterations')
ax5.set_ylabel(r'$R_\beta(f_\omega(X), X)$ loss')
plt.tight_layout()
fig5.savefig(save_fig_path + '/loss_orth.pdf')
