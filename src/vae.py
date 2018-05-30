# Variational auto-encoder for learning a representation for drum tracks
# James McDermott <jmmcd@jmmcd.net>
#
# Support for a paper submitted to CSMC 2018

# The VAE itself is based on
# https://raw.githubusercontent.com/pytorch/examples/master/vae/main.py

# standard library
from __future__ import print_function
import os
import itertools
import argparse

# numerical ecosystem
import matplotlib
matplotlib.use('agg') # avoid "invalid DISPLAY variable" on ssh
import matplotlib.pyplot as plt
import numpy as np

# torch
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# support module for midi-numpy conversion
from midi2numpy import save_numpy_as_midi



class GrooveMonkeeDataset(torch.utils.data.Dataset):
    """Groove Monkee MIDI drums dataset."""

    def __init__(self, filename, train=True, transform=None):
        # GM dataset is in range 0-127, so we divide.
        X = np.load(filename) / 127.

        # reshape to add an extra useless dimension since many
        # networks are set up to have 3 or 4 layers per image.
        X.shape = (X.shape[0], 1, X.shape[1],  X.shape[2])

        train_max = round(0.8 * X.shape[0])

        # shuffle deterministically: use seed 0. but save state first
        # and restore after.
        rng_state = np.random.get_state()
        np.random.seed(0)
        X = np.random.permutation(X)
        np.random.set_state(rng_state)

        
        if train:
            X = X[:train_max]
        else:
            X = X[train_max:]
                                         
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], "dummy label")


def load_data(dataset_name, args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if dataset_name == "MNIST":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        s0, s1 = 28, 28
        n_samples_to_plot = 10
    else:
        f = 'data/Groove_Monkee_Mega_Pack_AD.npy'
        train_loader = torch.utils.data.DataLoader(
            GrooveMonkeeDataset(f, train=True),
            batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(
            GrooveMonkeeDataset(f, train=False),
            batch_size=batch_size)
        s0, s1 = 9, 64
        n_samples_to_plot = 10

    return train_loader, test_loader, s0, s1, n_samples_to_plot




class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(s0 * s1, h)
        self.fc21 = nn.Linear(h, z) # mu
        self.fc22 = nn.Linear(h, z) # sigma
        # decoder
        self.fc3 = nn.Linear(z, h)
        self.fc4 = nn.Linear(h, s0 * s1)

    def encode(self, x):
        if activation == "ReLU":
            h1 = F.relu(self.fc1(x))
        elif activation == "tanh":
            h1 = F.tanh(self.fc1(x))
        else:
            raise ValueError
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # this function samples from the mean/variance to get a latent
        # code during training.  But during usage, there's no need for
        # variance, just use mean.
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            # don't add the noise during evaluation/use
            return mu

    def decode(self, z):
        if activation == "ReLU":
            h3 = F.relu(self.fc3(z))
            return F.sigmoid(self.fc4(h3))
        elif activation == "tanh":
            h3 = F.tanh(self.fc3(z))
            return F.tanh(self.fc4(h3))
        elif activation == "ReLU ReLU":
            h3 = F.relu(self.fc3(z))
            return F.relu(self.fc4(h3))
        else:
            raise ValueError

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, s0 * s1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and
# batch
def loss_function(recon_x, x, mu, logvar):

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    if loss_type == "BCE":
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, s0 * s1), size_average=False)
        return BCE, KLD
        
    elif loss_type == "MSE":
        MSE = F.mse_loss(recon_x, x.view(-1, s0 * s1), size_average=False)
        return MSE, KLD
    
    else:
        raise ValueError(loss_type)



def train(epoch):
    model.train() # this tells the model that we are in training mode
    train_loss = 0
    recon_loss = 0
    kl_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        recon_batch = recon_batch.clamp(min=0, max=1)
        _recon_loss, _kl_loss = loss_function(recon_batch, data, mu, logvar)
        loss = _recon_loss + _lambda * _kl_loss # _lambda to balance two terms
        loss.backward()
        recon_loss += _recon_loss.item()
        kl_loss    += _kl_loss.item()
        train_loss += loss.item()
        optimizer.step()
    recon_loss /= len(train_loader.dataset)
    kl_loss    /= len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    return train_loss, recon_loss, kl_loss

def test(epoch):
    model.eval() # tell the model we are not in training mode
    test_loss = 0
    recon_loss = 0
    kl_loss = 0
    with torch.no_grad(): # no need to track gradients in this block
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            # print(data.shape)
            recon_batch, mu, logvar = model(data)
            recon_batch = recon_batch.clamp(min=0, max=1)
            _recon_loss, _kl_loss = loss_function(recon_batch, data, mu, logvar)
            loss = _recon_loss + _lambda * _kl_loss # _lambda to balance two terms
            recon_loss += _recon_loss.item()
            kl_loss    += _kl_loss.item()
            test_loss += loss.item()

            if i == 0 and args.save_everything:
                plot_reconstruction(epoch, data, recon_batch)
        if args.save_everything:
            plot_bottleneck(epoch)
            plot_samples(epoch)

    recon_loss /= len(test_loader.dataset)
    kl_loss    /= len(test_loader.dataset)
    test_loss  /= len(test_loader.dataset)
    return test_loss, recon_loss, kl_loss




def plot_samples(epoch):
    sample = torch.randn(n_samples_to_plot, z).to(device)
    sample = model.decode(sample).cpu()
    X = 1.0 - sample.view(n_samples_to_plot, 1, s0, s1)
    save_image(X,
               basedir + 'sample_' + str(epoch).zfill(5) + '.png',
               nrow=1)

    np.save(basedir + 'sample_' + str(epoch).zfill(5) + '.npy', X)
    for i, Xi in enumerate(X):
        save_numpy_as_midi(basedir + 'sample_' + str(epoch).zfill(5) + '_' + str(i) + '.midi', 1.0 - Xi.numpy().squeeze(0))

    
def plot_reconstruction(epoch, data, recon):
    n = min(data.size(0), 8)
    comparison = torch.cat([data[:n],
                            recon.view(batch_size, 1, s0, s1)[:n]])
    X = 1.0 - comparison.cpu()
    save_image(X,
               basedir + 'reconstruction_' + str(epoch).zfill(5) + '.png', nrow=n)

    np.save(basedir + 'reconstruction_' + str(epoch).zfill(5) + '.npy', X)
    for i, Xi in enumerate(X):
        save_numpy_as_midi(basedir + 'reconstruction_' + str(epoch).zfill(5) + '_' + str(i) + '.midi', 1.0 - Xi.numpy().squeeze(0))
    
    
def plot_bottleneck(epoch):

    Zs = []
    for i, (data, _) in enumerate(test_loader):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        # we are going to look at mu only, not samples from the
        # Gaussians centered at mu
        _Z = mu.data.cpu().numpy() # argh, the conversion is a pain.
        Zs.append(_Z)
    Z = np.concatenate(Zs)

    # look at first two dimensions only
    plt.plot(Z[:, 0], Z[:, 1], ".")
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.axes().set_aspect('equal')
    plt.savefig(basedir + "bottleneck_{}.png".format(str(epoch).zfill(5)),
                bbox_inches='tight')
    plt.axis('equal')
    plt.close()

    np.save(basedir + "bottleneck_{}.npy".format(str(epoch).zfill(5)), Z)

def save_interpolations(epoch):
    sample = torch.zeros(n_samples_to_plot, z).to(device)
    sample[:, 0] = torch.linspace(-1, 1, n_samples_to_plot)
    sample = model.decode(sample).cpu()
    X = 1.0 - sample.view(n_samples_to_plot, 1, s0, s1)
    save_image(X,
               basedir + 'interpolation_' + str(epoch).zfill(5) + '.png',
               nrow=1)

    np.save(basedir + 'interpolation_' + str(epoch).zfill(5) + '.npy', X)
    for i, Xi in enumerate(X):
        save_numpy_as_midi(basedir + 'interpolation_' + str(epoch).zfill(5) + '_' + str(i) + '.midi', 1.0 - Xi.numpy().squeeze(0))
    





        
parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait before logging test status')
parser.add_argument('--save-everything', action='store_true', default=False,
                    help='save Numpy, MIDI, png and pdf outputs')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")


# hyperparameters. In our experiments, we test some variations.
# but in the code below, we run just a single configuration
# to make it easy and quick for user.
epochs = 300
z_vals = [2, 5, 10, 20, 40]
z_vals = [20]
h_vals = [100, 200, 400]
h_vals = [400]
batch_size_vals = [500]
lambda_vals = [0.0, 1.0]
lambda_vals = [1.0]
loss_type_vals = ["BCE", "MSE"]
loss_type_vals = ["BCE"]
activation_vals = ["ReLU", "tanh"]
activation_vals = ["ReLU"]

for z, h, batch_size, _lambda, loss_type, activation in \
    itertools.product(z_vals, h_vals, batch_size_vals,
                      lambda_vals, loss_type_vals,
                      activation_vals
    ):

    ks = "z_{}_h_{}_mb_{}_lambda_{}_loss_{}_act_{}/".format(
        z, h, batch_size, _lambda, loss_type, activation)
    basedir = "results/" + ks
    print(basedir)
    os.makedirs(basedir, exist_ok=True)
    ofile = open(basedir + "log.txt", "w")
    ofile.write("epoch, loss, recon_loss, kl_loss, test_loss, test_recon_loss, test_kl_loss\n")
    
    train_loader, test_loader, s0, s1, n_samples_to_plot = load_data("drums", args)
    
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        loss, recon_loss, kl_loss = train(epoch)
        if epoch % args.log_interval == 0:
            test_loss, test_recon_loss, test_kl_loss = test(epoch)
            s = "%4d %.2f %.2f %.2f %.2f %.2f %.2f" % (
                epoch, loss, recon_loss, kl_loss,
                test_loss, test_recon_loss, test_kl_loss)
            print(s)
            ofile.write(s + "\n")

    if args.save_everything:
        with torch.no_grad():
            save_interpolations(epochs)
            torch.save(model.state_dict(), basedir + 'model.pt')

