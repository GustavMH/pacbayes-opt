#!/usr/bin/env python3
import torch as T
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions import Normal


class LinearGaussian(nn.Module):
    """Linear linear where weights are sampled from a Gaussian every forward pass"""

    # https://medium.com/@pumplerod/probabilistic-neural-network-with-pytorch-11ec04479f67
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # No lower limit for variance is set
        # self.eps([1e-8])
        self.in_features = T.Tensor([in_features])
        self.out_features = T.Tensor([out_features])
        self.bias = bias

        # Kaiming-He init, fan in
        init_scale = T.sqrt(6 / self.in_features)
        layer_shape = (out_features, in_features)
        self.weight_mean = Parameter(T.rand(layer_shape) * init_scale)
        self.weight_std = Parameter(T.ones(layer_shape) * init_scale)
        if bias:
            self.bias_mean = Parameter(T.rand(out_features) * init_scale)
            self.bias_std = Parameter(T.ones(out_features) * init_scale)

    def forward(self, x: T.Tensor):
        weight = Normal(self.weight_mean, T.abs(self.weight_std)).rsample()
        bias = 0
        if self.bias:
            bias = Normal(self.bias_mean, T.abs(self.bias_std)).rsample()

        return x @ weight.T + bias



def KL_diag_gaussian(post_mean, post_std, prior_mean, prior_std):
    """Calculate the Kullbach-Leibler divergence between to multi-variate Gaussians with diagonal covariance matrices"""
    # https://arxiv.org/pdf/2102.05485v1
    assert post_mean.shape == post_std == prior_mean == prior_std

    n = post_mean.numel()
    prior_inv = 1 / prior_std
    return 0.5 * (
        T.log(T.norm(prior_mean, 2) / T.norm(post_mean, 2))
        + T.sum(post_std * prior_inv)  # trace
        + (prior_mean - post_mean).T * prior_inv * (prior_mean - post_mean)
        - n
    )


def upper_bound(
    empirical_error,
    weights_mean,
    weights_std,
    lambda_scale,
    dataset_size,
    delta=T.Tensor([0.025]),
    b=T.Tensor([100]),
    c=T.Tensor([0.1]),
):
    """ Calculate upper PAC-bayes bound with prior 0 """
    KL = KL_diag_gaussian(
        weights_mean,
        weights_std,
        T.zeros_like(weights_mean),
        lambda_scale * T.ones_like(weights_std),
    )

    B_RE = (
        KL
        + 2 * T.log(b * T.log(c / lambda_scale))
        + T.log((T.pi**2 * dataset_size) / (6 * delta))
    ) / (dataset_size - 1)

    return empirical_error + T.sqrt(0.5 * B_RE)

def train(model, dataset, loss_fn, optimizer, n_epochs, callbacks=[]):
    model.train()
    model.to(device)

    for epoch_n in tqdm(range(n_epochs)):
        for batch_n, (X, y) in enumerate(ds_loader(dataset, batch_size=1)): # Garbage code, FIXME
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

        for callback in callbacks:
            callback(model, epoch_n)

    return model

def ds_subset(ds, idx):
    ds_imgs, ds_labels = ds
    imgs = T.index_select(ds_imgs, 0, idx.to(device))
    labels = T.index_select(ds_labels, 0, idx.to(device))
    return (imgs, labels)

def ds_loader(ds, batch_size=64, shuffle=True):
    imgs, labels = ds
    perm = T.randperm(len(imgs)) if shuffle else T.arange(len(imgs))
    perm = perm.to(device)
    for i in range(0, len(imgs), batch_size):
        batch = perm[i : i + batch_size]
        yield (
            T.index_select(imgs, 0, batch),
            T.index_select(labels, 0, batch),
        )

from torchvision.datasets import MNIST
from torchvision.transforms import v2 as xform
from pathlib import Path
from tqdm import tqdm

SNN = nn.Sequential(LinearGaussian(28 * 28, 600), nn.ReLU(), LinearGaussian(600, 2))
binary_MNIST = MNIST(
    Path("./MNIST"),
    download=True,
    transform = xform.Compose([
        xform.PILToTensor(),
        xform.ToDtype(T.float32, 256),
        xform.Lambda(lambda x: x.flatten())
    ]),
    target_transform = lambda x: T.Tensor([1.0, 0.0] if x >= 5 else [0.0, 1.0])
)


if T.accelerator.is_available():
    device = T.accelerator.current_accelerator().type
else:
    device = "cpu"

#ds_labels = T.vstack([y for _, y in tqdm(binary_MNIST,"Load labels",ncols=20)]).to(device)
#ds_imgs = T.cat([X.unsqueeze(0) for X, _ in tqdm(binary_MNIST, "Load images", ncols=20)]).to(device)
ds = (ds_imgs, ds_labels)
opt = T.optim.SGD(SNN.parameters(), )
loss = nn.CrossEntropyLoss()
model = train(SNN, ds, loss, opt, 1)
