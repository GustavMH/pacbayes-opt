"""Sec. 4.2: train a deterministic MLP with SGD and keep its random initialization w0.

Run for example: python -m pacbayes.train_sgd --hidden 600
"""
import argparse
import os
import time

import torch

from pacbayes.data import load_binary_mnist
from pacbayes.networks import make_mlp, init_paper, get_weights, logistic_loss, zero_one_error
from pacbayes.utils import get_device, set_seed, experiment_name, save_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, nargs="+", default=[600], help="hidden layer sizes, e.g. 600")
    parser.add_argument("--epochs", type=int, default=20, help="20 for true labels, 120 for random labels (Sec. 4.2)")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--random-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-dir", default="mnist")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    name = experiment_name(args.hidden, args.random_labels)
    print("SGD | %s | %s | %s" % (name, device, vars(args)))

    train_x, train_y, test_x, test_y = load_binary_mnist(args.data_dir, args.random_labels, args.seed, device)
    m = train_x.shape[0]

    model = make_mlp(args.hidden)
    init_paper(model)
    model.to(device)
    w0 = get_weights(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    start_time = time.time()
    for epoch in range(args.epochs):
        permutation = torch.randperm(m, generator=shuffle_generator).to(device)
        for start in range(0, m, args.batch_size):
            batch = permutation[start:start + args.batch_size]
            loss = logistic_loss(model(train_x[batch]).squeeze(-1), train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        weights = get_weights(model)
        print("epoch %3d | train error %.4f | test error %.4f | %.0f s" % (
            epoch + 1, zero_one_error(weights, train_x, train_y), zero_one_error(weights, test_x, test_y),
            time.time() - start_time), flush=True)

    w_sgd = get_weights(model)
    result = {"experiment": name, "train_error": zero_one_error(w_sgd, train_x, train_y),
              "test_error": zero_one_error(w_sgd, test_x, test_y), "n_parameters": sum(w.numel() for w in w_sgd),
              "seconds": time.time() - start_time, "args": vars(args)}
    torch.save({"w0": [w.cpu() for w in w0], "w_sgd": [w.cpu() for w in w_sgd], "hidden": args.hidden,
                "random_labels": args.random_labels, "label_seed": args.seed},
               os.path.join(args.out_dir, "sgd_%s.pt" % name))
    save_json(result, os.path.join(args.out_dir, "sgd_%s.json" % name))
    print("SGD train error %.4f | test error %.4f | %d parameters" % (
        result["train_error"], result["test_error"], result["n_parameters"]))


if __name__ == "__main__":
    main()
