"""
A slightly modified example from here: https://github.com/optuna/optuna/blob/master/examples/pytorch_simple.py

Optuna example that optimizes multi-layer perceptrons using PyTorch.
"""


import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

"""
SAMPLING:
    we choose sampling strategy  to search our parameters,
    among other stratagies we could choose random seach, gird search etc.
    see documentaion for optuna.sampler

PRUNER:
    a strategy to prune experiments
    usually have following parametres as:

    FROM DOC.:
    - n_startup_trials – Pruning is disabled until the given number of trials finish in the same study.
    - n_warmup_steps – Pruning is disabled until the trial exceeds the given number of step.
      ( we need to get some median values first before we start to prune )
    - interval_steps – Interval in number of steps between the pruning checks, offset by the warmup steps.
     If no value has been reported at the time of a pruning check,
     that particular check will be postponed until a value is reported.


STUDY RESULTS
    are stored to sqlite, see below.
"""


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 20
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
STUDY_NAME = "TEST_MLP_PRUNING"


"""
To each function with parameters to optimize we pass 'trial' to suggest some parameters for us
there are several types of suggestions some of them will be below
"""

def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    # We pass name of a paramtre and some range of numbers to sample from
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_mnist():
    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    return train_loader, valid_loader


def evaluate_accuracy(model, valid_loader):
    # Validation of the model.
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            # Limiting validation data.
            if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                break
            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # Get the index of the max log-probability.
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / N_VALID_EXAMPLES
    return accuracy


def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    train_loader, valid_loader = get_mnist()

    # Training of the model.
    model.train()
    for epoch in range(EPOCHS):

        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)

            # Zeroing out gradient buffers.
            optimizer.zero_grad()
            # Performing a forward pass.
            output = model(data)
            # Computing negative Log Likelihood loss.
            loss = F.nll_loss(output, target)
            # Performing a backward pass.
            loss.backward()
            # Updating the weights.
            optimizer.step()

        acc = evaluate_accuracy(model, valid_loader)
        """
        If we want to prune / stop not performing experient early we should track intermediate values
        with trail.report(value, step) and optune will stop this trial.
        """
        trial.report(acc, epoch)
        if trial.should_prune(epoch):
            raise optuna.exceptions.TrialPruned()

    """
    Even though we track intermediate parameters we should return model's best score.
    Not necessary the last one, depends on your case.
    """
    return evaluate_accuracy(model, valid_loader)


if __name__ == "__main__":
    """
    Create a study and save stduy results to sqlite (for other storages see documentaion) 
    later we can laod our study to examine results (an example in jupyter notebook )
    study = optuna.study.load_study(study_name=STUDY_NAME, storage='sqlite:///example.db')
    """
    sampler = TPESampler(seed=10)
    study = optuna.create_study(
        direction="maximize",  # maximaze or minimaze our objective
        sampler=sampler,  # parametrs sampling strategy
        pruner=MedianPruner(
            n_startup_trials=15,
            n_warmup_steps=5,  # let's say num epochs
            interval_steps=2,
        ),
        study_name=STUDY_NAME,
        storage="sqlite:///example_4.db",  # storing study results, other storages are available too, see documentation.
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=50)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
