#!/usr/bin/env python3
import torch as T
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions import Normal


class LinearGaussian(nn.Module):
    """Linear linear where weights are sampled from a Gaussian every forward pass"""

    # https://medium.com/@pumplerod/probabilistic-neural-network-with-pytorch-11ec04479f67
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        # No lower limit for variance is set
        self.in_features = T.Tensor([in_features])
        self.out_features = T.Tensor([out_features])
        self.bias = bias

        # Kaiming-He init, fan in
        init_scale = T.sqrt(6 / self.in_features)
        layer_shape = (out_features, in_features)
        # TODO save initial weights unchanged for regularization
        self.weight_mean = Parameter(T.rand(layer_shape) * init_scale)
        self.weight_std = Parameter(T.abs(self.weight_mean) * init_scale)
        if bias:
            self.bias_mean = Parameter(T.rand(out_features) * init_scale)
            self.bias_std = Parameter(T.abs(self.weight_mean) * init_scale)

    def forward(self, x: T.Tensor):
        weight = Normal(self.weight_mean, T.exp(2*self.weight_std)).rsample()
        bias = 0
        if self.bias:
            bias = Normal(self.bias_mean, T.exp(2*self.bias_std)).rsample()

        return x @ weight.T + bias


def KL_diag_iso_gaussian(post_mean, post_std, lam):
    """Calculate the Kullbach-Leibler divergence between to multi-variate Gaussians with diagonal covariance matrices"""
    # https://arxiv.org/pdf/2102.05485v1
    assert post_mean.shape == post_std.shape

    d = post_mean.numel()
    return 0.5 * (
        (T.norm(post_std, 1) + T.norm(post_mean, 2) ** 2) / lam
        - d
        + d * T.log(lam)
        - T.sum(T.log(post_std))
    )


def train(model, dataset, loss_fn, optimizer, n_epochs, callbacks=[]):
    model.train()
    model.to(device)
    ds_size = len(dataset[0])

    for epoch_n in tqdm(range(n_epochs)):
        for batch_n, (X, y) in enumerate(
            ds_loader(dataset, batch_size=100)
        ):  # Garbage code, FIXME
            optimizer.zero_grad()
            pred = model(X)
            # constant to compensate for using BCEloss
            bce_mul = 1 / T.log(T.Tensor([2]))
            loss = loss_fn(pred, y) * bce_mul  + model.upper_bound_term(ds_size)

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


class SNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            LinearGaussian(28 * 28, 600), nn.ReLU(), LinearGaussian(600, 1)
        )
        self.lambda_constrained = nn.Parameter(T.Tensor([-3]))

    def forward(self, x):
        return self.model(x)

    def get_weights(self, query):
        return T.cat(
            [w.flatten() for name, w in self.named_parameters() if query in name]
        )

    def upper_bound_term(
        self,
        dataset_size,
        delta=T.Tensor([0.025]),
        b=T.Tensor([100]),
        c=T.Tensor([0.1]),
    ):
        """Calculate upper PAC-bayes bound with prior 0"""
        weights_mean = self.get_weights("_mean")
        weights_std = self.get_weights("_std")
        lambda_scale = T.exp(2 * self.lambda_constrained)

        KL = KL_diag_iso_gaussian(
            weights_mean,
            weights_std,
            lambda_scale
        )

        B_RE = (
            KL
            + 2 * T.log(b * T.log(c / lambda_scale))
            + T.log((T.pi**2 * dataset_size) / (6 * delta))
        ) / (dataset_size - 1)

        print()
        print(f"{T.norm(weights_std, 1)}")

        return T.sqrt(0.5 * B_RE)


binary_MNIST = MNIST(
    Path("./MNIST"),
    download=True,
    transform=xform.Compose(
        [
            xform.PILToTensor(),
            xform.ToDtype(T.float32, 256),
            xform.Lambda(lambda x: x.flatten()),
        ]
    ),
    target_transform=lambda x: T.Tensor([1 if x >= 5 else -1]),
)


if T.accelerator.is_available():
    device = T.accelerator.current_accelerator().type
else:
    device = "cpu"

#ds_labels = T.vstack([y for _, y in tqdm(binary_MNIST,"Load labels",ncols=20)]).to(device)
#ds_imgs = T.cat([X.unsqueeze(0) for X, _ in tqdm(binary_MNIST, "Load images", ncols=20)]).to(device)
ds = (ds_imgs, ds_labels)
snn = SNN()
opt = T.optim.RMSprop(snn.parameters(), lr=0.001)
loss = nn.BCEWithLogitsLoss()
model = train(snn, ds, loss, opt, 1)
