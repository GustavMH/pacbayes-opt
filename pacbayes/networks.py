"""Multilayer perceptrons (Sec. 4.2) and the losses of Sec. 2."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(hidden):
    """Fully connected network with ReLU at every hidden node and a linear output."""
    layers, n_in = [], 784
    for n_out in hidden:
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        n_in = n_out
    layers.append(nn.Linear(n_in, 1))
    return nn.Sequential(*layers)


def init_weights(model):
    """Weights ~ normal(0, 0.04) truncated to [-2*0.04, 2*0.04]
    biases 0.1 in the first layer, 0 elsewhere."""
    linear_layers = [m for m in model if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.04, a=-0.08, b=0.08)
        nn.init.constant_(layer.bias, 0.1 if i == 0 else 0.0)


def get_weights(model):
    """Parameters as a list [W1, b1, W2, b2, ...] with W of shape (in, out)."""
    weights = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            weights += [layer.weight.detach().t().contiguous().clone(), layer.bias.detach().clone()]
    return weights


def forward(weights, x):
    """Network output for weights [W1, b1, ...]. With a leading dimension K on every weight tensor, K networks are
    evaluated at once and the output has shape (K, N); otherwise it has shape (N,)."""
    h = x
    for i in range(0, len(weights), 2):
        if i > 0:
            h = torch.relu(h)
        W, b = weights[i], weights[i + 1]
        h = torch.matmul(h, W) + (b.unsqueeze(-2) if W.dim() == 3 else b)
    return h.squeeze(-1)


def logistic_loss(output, y):
    """(1 / log 2) log(1 + exp(-y * output)), an upper bound on the 0-1 loss (Sec. 2)."""
    return F.softplus(-y * output).mean() / math.log(2)


def predict(output):
    return torch.where(output >= 0, 1.0, -1.0)


def zero_one_error(weights, x, y, chunk_size=5500):
    """Fraction of examples with sign(output) != y."""
    wrong = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            wrong += (predict(forward(weights, x[start:start + chunk_size])) != y[start:start + chunk_size]).sum().item()
    return wrong / x.shape[0]
