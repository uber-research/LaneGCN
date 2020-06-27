import argparse
import os
import pickle
import random
import sys
import time
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import DataLoader

os.umask(0)


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(
    description="Data preprocess for argo forcasting dataset"
)
parser.add_argument(
    "-m", "--model", default="model", type=str, metavar="MODEL", help="model name"
)


def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    print(args.model)
    config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()

    config["preprocess"] = False  # we use raw data to generate preprocess data
    config["val_workers"] = 32
    config["workers"] = 32

    # Data loader for training set
    dataset = Dataset(config["train_split"], config, train=True)
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    # Data loader for validation set
    dataset = Dataset(config["val_split"], config, train=False)
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Data loader for test set
    dataset = Dataset(config["test_split"], config, train=False)
    test_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    train(config, train_loader, args.model)
    val(config, val_loader, args.model)
    test(config, test_loader, args.model)


def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train(config, train_loader, model):
    stores = [None for x in range(205942)]

    t = time.time()
    for i, data in enumerate(train_loader):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    f = open(config["preprocess_train"], "wb")
    pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def val(config, data_loader, model):
    stores = [None for x in range(39472)]

    t = time.time()
    for i, data in enumerate(data_loader):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    f = open(config["preprocess_val"], "wb")
    pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def test(config, data_loader, model):
    stores = [None for x in range(78143)]

    t = time.time()
    for i, data in enumerate(data_loader):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "city",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "rot",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    f = open(config["preprocess_test"], "wb")
    pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def to_numpy(data):
    """Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data


def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data


if __name__ == "__main__":
    main()
