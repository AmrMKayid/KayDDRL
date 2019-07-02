import torch
from torch import optim

from kayddrl.agents.algos import DDPGAgent
from kayddrl.comp.ounoise import OUNoise
from kayddrl.configs.default import Configs
from kayddrl.envs.base import BaseEnv
from kayddrl.networks.mlp import MLP


def run(env: BaseEnv, configs: Configs, state_dim: int, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """

    device = configs.glob.device

    hidden_sizes_actor = [256, 256]
    hidden_sizes_critic = [256, 256]

    # create actor
    actor = MLP(
        input_size=state_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_actor,
        output_activation=torch.tanh,
    ).to(device)

    actor_target = MLP(
        input_size=state_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_actor,
        output_activation=torch.tanh,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # create critic
    critic = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic_target = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())

    # create optimizer
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=configs.models.lr_actor,
        weight_decay=configs.models.weight_decay,
    )

    critic_optim = optim.Adam(
        critic.parameters(),
        lr=configs.models.lr_critic,
        weight_decay=configs.models.weight_decay,
    )

    # noise
    noise = OUNoise(
        action_dim,
        theta=configs.agent.ou_noise_theta,
        sigma=configs.agent.ou_noise_sigma,
    )

    # make tuples to create an agent
    models = (actor, actor_target, critic, critic_target)
    optims = (actor_optim, critic_optim)

    # create an agent
    agent = DDPGAgent(env, models, optims, noise)

    # run
    if configs.glob.test:
        agent.test()
    else:
        agent.train()
