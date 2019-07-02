from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def identity(x: torch.Tensor) -> torch.Tensor:
    """Return input without any change."""
    return x


def make_one_hot(labels: torch.Tensor, c: int, device):
    """Converts an integer label to a one-hot Variable."""
    y = torch.eye(c).to(device)
    labels = labels.type(torch.LongTensor)
    return y[labels]


def concat(
        in_1: torch.Tensor, in_2: torch.Tensor, n_category: int = -1
) -> torch.Tensor:
    """Concatenate state and action tensors properly depending on the action."""
    in_2 = make_one_hot(in_2, n_category) if n_category > 0 else in_2

    if len(in_2.size()) == 1:
        in_2 = in_2.unsqueeze(0)

    in_concat = torch.cat((in_1, in_2), dim=-1)

    return in_concat


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer"""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class MLP(nn.Module):
    """Baseline of Multilayer perceptron.

    Attributes:
        input_size (int): size of input
        output_size (int): size of output layer
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers
        use_output_layer (bool): whether or not to use the last layer
        n_category (int): category number (-1 if the action is continuous)

    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_sizes: list,
            hidden_activation: Callable = F.relu,
            output_activation: Callable = identity,
            linear_layer: nn.Module = nn.Linear,
            use_output_layer: bool = True,
            n_category: int = -1,
            init_fn: Callable = init_layer_uniform,
    ):
        """Initialization.

        Args:
            input_size (int): size of input
            output_size (int): size of output layer
            hidden_sizes (list): number of hidden layers
            hidden_activation (function): activation function of hidden layers
            output_activation (function): activation function of output layer
            linear_layer (nn.Module): linear layer of mlp
            use_output_layer (bool): whether or not to use the last layer
            n_category (int): category number (-1 if the action is continuous)
            init_fn (Callable): weight initialization function bound for the last layer

        """
        super(MLP, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.linear_layer = linear_layer
        self.use_output_layer = use_output_layer
        self.n_category = n_category

        # set hidden layers
        self.hidden_layers: list = []
        in_size = self.input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = self.linear_layer(in_size, next_size)
            in_size = next_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # set output layers
        if self.use_output_layer:
            self.output_layer = self.linear_layer(in_size, output_size)
            self.output_layer = init_fn(self.output_layer)
        else:
            self.output_layer = identity
            self.output_activation = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))

        return x
