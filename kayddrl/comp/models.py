import torch
import torch.nn as nn
import pydash as ps

from functools import partial, wraps

from kayddrl.utils import utils


# -------------------- Fully Connected -------------------- #

def get_nn_name(name):
    for nn_name in nn.__dict__:
        if name.lower() == nn_name.lower():
            return nn_name
    raise ValueError(f'Name {name} not found in {nn.__dict__}')


def get_activation_fn(activation):
    ActivationClass = getattr(nn, get_nn_name(activation))
    return ActivationClass()


def make_fc(dims, activation=None):
    assert len(dims) >= 2, 'dims need to at least contain input, output'
    pairs = list(zip(dims[:-1], dims[1:]))
    layers = []
    for in_d, out_d in pairs:
        layers.append(nn.Linear(in_d, out_d))
        if activation is not None:
            layers.append(get_activation_fn(activation))
    model = nn.Sequential(*layers)
    return model


# -------------------- ---------- -------------------- #

# -------------------- Init -------------------- #


def init_params(module, init_fn):
    bias_init = 0.0
    classname = utils.get_cls_name(module)

    if 'Net' in classname:
        pass

    elif any(k in classname for k in ('BatchNorm', 'Conv', 'Linear')):
        init_fn(module.weight)
        nn.init.constant_(module.bias, bias_init)

    elif 'GRU' in classname:
        for name, param in module.named_parameters():
            if 'weight' in name:
                init_fn(param)
            elif 'bias' in name:
                nn.init.constant_(param, bias_init)
    else:
        pass


def init_layers(net, init_fn_name):
    if init_fn_name is None:
        return

    nonlinearity = get_nn_name(net.hid_layers_activation).lower()
    if nonlinearity == 'leakyrelu':
        nonlinearity = 'leaky_relu'

    init_fn = getattr(nn.init, init_fn_name)

    if 'kaiming' in init_fn_name:
        assert nonlinearity in ['relu', 'leaky_relu'], f'Kaiming initialization not supported for {nonlinearity}'
        init_fn = partial(init_fn, nonlinearity=nonlinearity)

    elif 'orthogonal' in init_fn_name or 'xavier' in init_fn_name:
        gain = nn.init.calculate_gain(nonlinearity)
        init_fn = partial(init_fn, gain=gain)

    else:
        pass

    net.apply(partial(init_params, init_fn=init_fn))


# -------------------- ---------- -------------------- #

# -------------------- Get -------------------- #

def get_loss_fn(cls, loss_config):
    LossClass = getattr(nn, get_nn_name(loss_config['name']))
    loss_config = ps.omit(loss_config, 'name')
    loss_fn = LossClass(**loss_config)
    return loss_fn

# -------------------- ---------- -------------------- #
