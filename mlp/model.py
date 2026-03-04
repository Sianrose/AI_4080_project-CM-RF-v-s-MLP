"""
Configurable Multi-Layer Perceptron for Sign Language MNIST.

Used by train_mlp.py and app/app.py. Keeps the model definition
in one place so training and inference always match.
"""

import torch.nn as nn

NUM_CLASSES = 24
INPUT_SIZE = 784  # 28×28 flattened


def _get_activation(name):
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(0.1),
    }
    return activations.get(name, nn.ReLU())


class SignLanguageMLP(nn.Module):
    """
    MLP with configurable hidden layers, activation, and dropout.
    No batch normalization (Round 1 model).
    """

    def __init__(self, hidden_sizes, activation="relu", dropout=0.0):
        super().__init__()
        act_fn = _get_activation(activation)

        layers = []
        in_size = INPUT_SIZE
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = h_size

        layers.append(nn.Linear(in_size, NUM_CLASSES))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SignLanguageMLP_v2(nn.Module):
    """
    Improved MLP with Batch Normalization after each linear layer.
    BatchNorm stabilizes training and typically boosts accuracy by 5–10%.
    """

    def __init__(self, hidden_sizes, activation="relu", dropout=0.0):
        super().__init__()
        act_fn = _get_activation(activation)

        layers = []
        in_size = INPUT_SIZE
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = h_size

        layers.append(nn.Linear(in_size, NUM_CLASSES))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
