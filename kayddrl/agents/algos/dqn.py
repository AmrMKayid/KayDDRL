# import torch
# import numpy as np
#
# from kayddrl import networks
# from kayddrl.agents.algos.base import BaseAlgo
# from kayddrl.comp import policies, models, scheduler
# from kayddrl.utils import utils
# from kayddrl.utils.logging import logger
#
#
# class DQN(BaseAlgo):
#
#     def __init__(self, agent, config):
#         super(DQN, self).__init__(agent, config)
#
#     def _build(self):
#         self.init_hparams()
#         self.init_net()
#
#     def act(self, state):
#         '''Note, SARSA is discrete-only'''
#         agent = self._agent
#         action = self.action_policy(state, self, agent)
#         return action.cpu().squeeze().numpy()  # squeeze to handle scalar
#
#     def step(self, state, action, reward, next_state, done):
#         self._agent.memory.update(state, action, reward, next_state, done)
#
#         # Learn every update_every time steps.
#         self.t_step = (self.t_step + 1) % self.update_every
#         if self.t_step == 0:
#             # If enough samples are available in memory, get random subset and learn
#             if len(self._agent.memory) > self._agent.memory.batch_size:
#                 experiences = self._agent.memory.sample()
#                 self.learn(experiences)
#
#     def learn(self, experiences, indexes=None):
#         """Update value parameters using given batch of experience tuples.
#
#         Params
#         ======
#             experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
#             gamma (float): discount factor
#         """
#
#         states, actions, rewards, next_states, dones = experiences
#
#         # get q values from local model
#         q_out = self.net(states)
#
#         # get q values for chosen action
#         q_a_predictions = q_out.gather(1, actions.long().unsqueeze(-1))
#
#         max_q_targets_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
#
#         # calculate td targets
#         targets = rewards + (self.gamma * max_q_targets_next * (1 - dones))
#
#         # calculate loss
#         loss = self.net.loss_fn(q_a_predictions, targets)
#
#         self.net.train_step(loss, self.optim, self.lr_scheduler)
#
#         # update stats
#         with torch.no_grad():
#             self._agent.loss_list.append(loss.item())
#             # calculate sparse softmax cross entropy
#             # self._agent.entropy_list.append(F.cross_entropy(q_out, actions.squeeze(1)))
#
#         logger.warn(
#             'Learning: \t\t Memory Size: {} \t | \t Loss = {} \t | \t mean loss = {}'.format(len(self._agent.memory),
#                                                                                              loss,
#                                                                                              np.mean(
#                                                                                                  self._agent.loss_list)))
#
#         # update target network
#         self.to_update_nets = (self.to_update_nets + 1) % self.net.update_frequency
#         if self.to_update_nets == 0:
#             self.update_nets()
#
#     def update_nets(self):
#         if self.net.update_type == 'replace':
#             models.copy(self.net, self.target_net)
#         elif self.net.update_type == 'polyak':
#             models.polyak_update(self.net, self.target_net, self.net.tau_polyak_coef)
#         else:
#             raise ValueError('Unknown net.update_type. Should be "replace" or "polyak". Exiting.')
#         logger.info('Target net is updated!')
#
#     def init_hparams(self):
#         utils.set_attr(self, dict(
#             t_step=0,  # Initialize time step (for updating every update_every steps)
#             gamma=0.99,  # discount factor
#             update_every=4,  # how often to update the network
#             to_update_nets=0,  # Initialize to update nets (for updating every update_every steps)
#             epsilon_config=None,
#             action_prob_dist_type='Argmax',
#             action_policy='epsilon_greedy',
#         ))
#
#         utils.set_attr(self, self._config, [
#             'gamma',
#             'update_every',
#             'epsilon_config',
#             'action_prob_dist_type',
#             'action_policy',
#         ])
#
#         self.action_policy = getattr(policies, self.action_policy)
#         self.epsilon_scheduler = scheduler.VarScheduler(self.epsilon_config)
#         self.epsilon = self.epsilon_scheduler.start_val
#
#     def init_net(self):
#         input_state_size = self._agent.state_dim
#         output_action_size = models.get_out_dim(self._agent)
#         NetClass = getattr(networks, self.net_config['type'])
#         self.net = NetClass(self.net_config, input_state_size, output_action_size)
#         self.target_net = NetClass(self.net_config, input_state_size, output_action_size)
#         self.net_names = ['net', 'target_net']
#
#         self.optim = models.get_optim(self.net, self.net.optim_config)
#         self.lr_scheduler = models.get_lr_scheduler(self.optim, self.net.lr_scheduler_config)
#
#     def calc_prob_dist_param(self, x, net=None):
#         r"""
#         To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the correct outputs.
#         The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
#         :param x:
#         :param net:
#         :return:
#         """
#         net = self.net if net is None else net
#         prob_dist_param = net(x)
#         return prob_dist_param
#
#     def save(self, ckpt=None):
#         pass
#
#     def load(self):
#         pass
