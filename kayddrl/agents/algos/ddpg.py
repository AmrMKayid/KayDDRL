import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kayddrl.agents.base import BaseAgent
from kayddrl.comp.ounoise import OUNoise
from kayddrl.configs.default import Configs, default
from kayddrl.envs.base import BaseEnv
from kayddrl.memory import ReplayBuffer
from kayddrl.utils.logging import logger
from kayddrl.utils.utils import describe


class DDPGAgent(BaseAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (ReplayBuffer): replay memory
        noise (OUNoise): random noise for exploration
        hyper_params (dict): hyper-parameters
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step numbers
        episode_step (int): step number of the current episode
        i_episode (int): current episode number

    """

    def __init__(self, env: BaseEnv, models: tuple, optims: tuple, noise: OUNoise, configs: Configs = default()):

        super(DDPGAgent, self).__init__(env, configs)

        self.actor, self.actor_target, self.critic, self.critic_target = models
        self.actor_optimizer, self.critic_optimizer = optims
        self.curr_state = np.zeros((1,))
        self.noise = noise
        self.total_step = 0
        self.episode_step = 0
        self.i_episode = 0

        if configs.glob.load_from is not None and os.path.exists(configs.glob.load_from):
            self.load_params(configs.glob.load_from)

        self._initialize()

        logger.info(describe(self))

    def _initialize(self):
        """Initialize non-common things."""
        if not self._configs.glob.test:
            # replay memory
            self.memory = ReplayBuffer(self._configs)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state
        state = self._preprocess_state(state)

        # if initial random action should be conducted
        if (
                self.total_step < self._hparams.initial_random_action
                and not self._configs.glob.test
        ):
            return self._env.action_space.sample()

        selected_action = self.actor(state).detach().cpu().numpy()

        if not self._configs.glob.test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        return selected_action

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        state = torch.FloatTensor(state).to(self._configs.glob.device)
        return state

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, ...]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self._env.step(action)

        if not self._configs.glob.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self._configs.glob.max_episode_steps else done
            )
            transition = (self.curr_state, action, reward, next_state, done_bool)
            self.memory.update(*transition)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(torch.cat((next_states, next_actions), dim=-1))
        curr_returns = rewards + self._hparams.gamma * next_values * masks
        curr_returns = curr_returns.to(self._configs.glob.device)

        # train critic
        gradient_clip_cr = self._hparams.gradient_clip_cr
        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss = F.mse_loss(values, curr_returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), gradient_clip_cr)
        self.critic_optimizer.step()

        # train actor
        gradient_clip_ac = self._hparams.gradient_clip_ac
        actions = self.actor(states)
        actor_loss = -self.critic(torch.cat((states, actions), dim=-1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), gradient_clip_ac)
        self.actor_optimizer.step()

        # update target networks
        tau = self._hparams.tau
        self.soft_update(self.actor, self.actor_target, tau)
        self.soft_update(self.critic, self.critic_target, tau)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, local: nn.Module, target: nn.Module, tau: float):
        """Soft-update: target = tau*local + (1-tau)*target."""
        for t_param, l_param in zip(target.parameters(), local.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            logger.fatal("the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.actor_target.load_state_dict(params["actor_target_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.critic_target.load_state_dict(params["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optimizer.load_state_dict(params["critic_optim_state_dict"])
        logger.info("loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optim_state_dict": self.actor_optimizer.state_dict(),
            "critic_optim_state_dict": self.critic_optimizer.state_dict(),
        }

        BaseAgent.save_params(self, params, n_episode)

    def write_log(self, i: int, loss: np.ndarray, score: int, avg_score):
        """Write log about loss and score"""
        total_loss = loss.sum()

        logger.info(
            "episode %d:\t episode step: %d | total step: %d | total score: %d |\t"
            "total loss: %f | actor_loss: %.3f | critic_loss: %.3f\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0],
                loss[1],
            )  # actor loss  # critic loss
        )

        if self._configs.glob.log:
            logger.log_scalar("scores/score", score, i)
            logger.log_scalar("scores/avg_score", avg_score, i)
            logger.log_scalar("losses/total_loss", total_loss, i)
            logger.log_scalar("losses/actor_loss", loss[0], i)
            logger.log_scalar("losses/critic_loss", loss[1], i)

    def train(self):
        """Train the agent."""
        logger.warn("Start training")

        for self.i_episode in range(1, self._configs.glob.num_episodes + 1):
            state = self._env.reset()
            done = False
            score = 0
            total_score = []
            self.episode_step = 0
            losses = list()

            while not done:
                self._env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                self.total_step += 1
                self.episode_step += 1
                self._configs.glob.global_step += 1

                if len(self.memory) >= self._configs.memory.batch_size:
                    for _ in range(self._hparams.multiple_learn):
                        loss = self.update_model()
                        losses.append(loss)  # for logging

                state = next_state
                score += reward

            # logging
            if losses:
                total_score.append(score)
                avg_loss = np.vstack(losses).mean(axis=0)
                self.write_log(self.i_episode, avg_loss, score, np.mean(total_score))
                losses.clear()

            if self.i_episode % self._configs.glob.save_period == 0:
                self.save_params(self.i_episode)
                self.interim_test()

        # termination
        self._env.close()
        self.save_params(self.i_episode)
        self.interim_test()

# ddpg_agent = DDPGAgent(None, (1,2,3,4), (1,2), None)
