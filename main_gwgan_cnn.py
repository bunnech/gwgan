#!/usr/bin/python
# author: Charlotte Bunne

# imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import os
from time import time
from torchvision import datasets, transforms
from torchvision.utils import save_image

# internal imports
from model.utils import *
from model.model_cnn import Generator, Adversary
from model.model_cnn import weights_init_generator, weights_init_adversary
from model.loss import gwnorm_distance, loss_total_variation, loss_procrustes

# get arguments
args = get_args()

# system preferences
seed = np.random.randint(100)
torch.set_default_dtype(torch.double)
np.random.seed(seed)
torch.manual_seed(seed)

# settings
batch_size = 256
z_dim = 100
lr = 0.0002
ngen = 3
beta = args.beta
lam = 0.5
niter = 10
epsilon = 0.005
num_epochs = args.num_epochs
cuda = args.cuda
channels = args.n_channels
id = args.id

model = 'gwgan_{}_eps_{}_tv_{}_procrustes_{}_ngen_{}_channels_{}_id_{}' \
        .format(args.data, epsilon, lam, beta, ngen, channels, id)
save_fig_path = 'out_' + model
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)

# data import
if args.data == 'mnist':
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=batch_size, drop_last=True, shuffle=True)
elif args.data == 'fmnist':
    dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data/fmnist', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(32),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))])),
        batch_size=batch_size, drop_last=True, shuffle=True)
elif args.data == 'cifar_gray':
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data/cifar10', train=True, download=True,
                         transform=transforms.Compose([
                            # transform RGB to grayscale
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))])),
        batch_size=batch_size, drop_last=True, shuffle=True)
elif args.data == 'cifar':
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data/cifar10', train=True, download=True,
                         transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))])),
        batch_size=batch_size, drop_last=True, shuffle=True)
else:
    raise NotImplementedError('dataset does not exist or not integrated.')

# print example images
save_image(next(iter(dataloader))[0][:25],
           os.path.join(save_fig_path, 'real.pdf'), nrow=5, normalize=True)

# define networks and parameters
generator = Generator(output_dim=channels)
adversary = Adversary(input_dim=channels)

# weight initialisation
generator.apply(weights_init_generator)
adversary.apply(weights_init_adversary)

if cuda:
    generator = generator.cuda()
    adversary = adversary.cuda()

# create optimizer
g_optimizer = torch.optim.Adam(generator.parameters(), lr, betas=(0.5, 0.99))
# zero gradients
generator.zero_grad()

c_optimizer = torch.optim.Adam(adversary.parameters(), lr, betas=(0.5, 0.99))
# zero gradients
adversary.zero_grad()

# sample for plotting
num_test_samples = batch_size
z_ex = torch.randn(num_test_samples, z_dim)
if cuda:
    z_ex = z_ex.cuda()

loss_history = list()
loss_tv = list()
loss_orth = list()
loss_og = 0
is_hist = list()

for epoch in range(num_epochs):
    t0 = time()

    for it, (image, _) in enumerate(dataloader):
        train_c = ((it + 1) % (ngen + 1) == 0)

        x = image.double()
        if cuda:
            x = x.cuda()

        # sample random number z from Z
        z = torch.randn(image.shape[0], z_dim)

        if cuda:
            z = z.cuda()

        if train_c:
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
        loss, T = gwnorm_distance((D_x, D_x_norm), (D_g, D_g_norm),
                                  epsilon, niter, loss_fun='square_loss',
                                  coupling=True, cuda=cuda)

        if train_c:
            # train adversary
            loss_og = loss_procrustes(f_x, x.view(x.shape[0], -1), cuda)
            loss_to = -loss + beta * loss_og
            loss_to.backward()

            # parameter updates
            c_optimizer.step()
            # zero gradients
            reset_grad(generator, adversary)

        else:
            # train generator
            loss_t = loss_total_variation(g)
            loss_to = loss + lam * loss_t
            loss_to.backward()

            # parameter updates
            g_optimizer.step()
            # zero gradients
            reset_grad(generator, adversary)

    # plotting
    # get generator example
    g_ex = generator.forward(z_ex)
    g_plot = g_ex.cpu().detach()

    # plot result
    save_image(g_plot.data[:25],
               os.path.join(save_fig_path, 'g_%d.pdf' % epoch),
               nrow=5, normalize=True)

    fig1, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax0 = ax[0].imshow(T.cpu().detach().numpy(), cmap='RdBu_r')
    colorbar(ax0)
    ax1 = ax[1].imshow(D_x.cpu().detach().numpy(), cmap='Blues')
    colorbar(ax1)
    ax2 = ax[2].imshow(D_g.cpu().detach().numpy(), cmap='Blues')
    colorbar(ax2)
    ax[0].set_title(r'$T$')
    ax[1].set_title(r'inner distances of $D$')
    ax[2].set_title(r'inner distances of $G$')
    plt.tight_layout(h_pad=1)
    fig1.savefig(os.path.join(save_fig_path, '{}_ccc.pdf'.format(
            str(epoch).zfill(3))), bbox_inches='tight')

    loss_history.append(loss)
    loss_tv.append(loss_t)
    loss_orth.append(loss_og)
    plt.close('all')

# plot loss history
fig2 = plt.figure(figsize=(2.4, 2))
ax2 = fig2.add_subplot(111)
ax2.plot(loss_history, 'k.')
ax2.set_xlabel('Iterations')
ax2.set_ylabel(r'$\overline{GW}_\epsilon$ Loss')
plt.tight_layout()
plt.grid()
fig2.savefig(save_fig_path + '/loss_history.pdf')

fig3 = plt.figure(figsize=(2.4, 2))
ax3 = fig3.add_subplot(111)
ax3.plot(loss_tv, 'k.')
ax3.set_xlabel('Iterations')
ax3.set_ylabel(r'Total Variation Loss')
plt.tight_layout()
plt.grid()
fig3.savefig(save_fig_path + '/loss_tv.pdf')

fig4 = plt.figure(figsize=(2.4, 2))
ax4 = fig4.add_subplot(111)
ax4.plot(loss_orth, 'k.')
ax4.set_xlabel('Iterations')
ax4.set_ylabel(r'$R_\beta(f_\omega(X), X)$ Loss')
plt.tight_layout()
plt.grid()
fig4.savefig(save_fig_path + '/loss_orth.pdf')
