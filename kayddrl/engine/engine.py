import torch

from kayddrl.utils import utils
from kayddrl.utils.logging import logger


class Engine:

    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self._config = config
        self._logger = logger
        self.max_time_steps = env.max_steps
        self.score = 0.0

        utils.set_attr(self, self._config, [
            'num_episodes',
            'to_be_solved_score'
        ])

    def train(self):

        stats_format = 'ε: {:6.4f}  ⍺: {:6.4f}  Buffer: {:6}'

        for i_episode in range(1, self.num_episodes + 1):
            rewards = []
            state = self.env.reset()

            # loop over steps
            for t in range(1, self.max_time_steps + 1):
                self.env.render()

                # select an action
                action = self.agent.act(state)

                next_state, reward, done, info = self.env.step(action)

                # update agent with returned information
                self.agent.observe(state, action, reward, next_state, done)

                state = next_state
                rewards.append(reward)

                if done:
                    break

        #     # every episode
        #     # eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        #     buffer_len = len(self.agent.memory)
        #     self.logger.update(t, rewards, i_episode)
        #     self.logger.print_epi(i_episode, t, stats_format)  # , eps, agent.alpha, buffer_len)
        #
        #     # every epoch (100 episodes)
        #     if i_episode % 100 == 0:
        #         # self.logger.print_epoch(i_episode, stats_format)  # , eps, agent.alpha, buffer_len)
        #         save_name = 'checkpoints/last_run/episode.{}'.format(i_episode)
        #         torch.save(self.agent.algo.net.state_dict(), save_name + '.pth')
        #         print('saved model')
        #         # dill.dump(agent.memory, open(save_name + '.buffer.pck', 'wb'))
        #
        #     # if solved
        #     if self.logger.is_solved(i_episode, self.to_be_solved_score):
        #         self.logger.print_solved(i_episode, stats_format)  # , eps, agent.alpha, buffer_len)
        #         save_name = 'checkpoints/last_run/solved'
        #         torch.save(self.agent.qnetwork_local.state_dict(), save_name + '.pth')
        #         # dill.dump(agent.memory, open(save_name + '.buffer.pck', 'wb'))
        #         break
        #
        # # # training finished
        # # if graph_when_done:
        # #     self.logger.plot(agent.loss_list, agent.entropy_list)

        self.env.close()
