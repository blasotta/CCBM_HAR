import torch
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from maf import MAF

from utils.validation import val_maf
from utils.test import test_maf
from utils.plot import sample_digits_maf
from datasets.data_loaders import get_data_loaders, get_data

string = "maf_mnist_512"
dataset = "mnist"
batch_size = 128

model = torch.load("model_saves/" + string + ".pt")
data = get_data(dataset)
train = torch.from_numpy(data.train.x)
train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
test_maf(model, train, test_loader)
val_maf(model, train, val_loader)
#sample_digits_maf(model, "test")

if dataset == "mnist":
    if not os.path.exists("figs"):
        os.makedirs("figs")
    _, _, test_loader = get_data_loaders(data, batch_size=1000)
    model.eval()
    batch = next(iter(test_loader))
    u = model(batch)[0].detach().numpy()
    fig, axes = plt.subplots(
        ncols=6, nrows=4, sharex=True, sharey=True, figsize=(16, 10)
    )

    for ax in axes.reshape(-1):
        dim1 = np.random.randint(28 * 28)
        dim2 = np.random.randint(28 * 28)
        ax.scatter(u[:, dim1], u[:, dim2], color="dodgerblue", s=0.5)
        ax.set_ylabel("dim: " + str(dim2), size=14)
        ax.set_xlabel("dim: " + str(dim1), size=14)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect(1)

    plt.savefig("figs/" + string + "_scatter.png", bbox_inches="tight", dpi=300)
    plt.savefig("figs/" + string + "_scatter.pdf", bbox_inches="tight", dpi=300)

    fig, axes = plt.subplots(
        ncols=6, nrows=4, sharex=True, sharey=True, figsize=(16, 10)
    )

    for ax in axes.reshape(-1):
        dim1 = np.random.randint(28 * 28)
        sns.histplot(u[:, dim1], ax=ax, color="darkorange")
        ax.set_xlabel("dim: " + str(dim1), size=14)
        ax.set_xlim(-5, 5)

    plt.savefig("figs/" + string + "_marginal.png", bbox_inches="tight", dpi=300)
    plt.savefig("figs/" + string + "_marginal.pdf", bbox_inches="tight", dpi=300)

    sample_digits_maf(model, "test")

