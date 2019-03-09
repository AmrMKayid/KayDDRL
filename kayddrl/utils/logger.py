import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from kayddrl.utils.viz import subplot


class Logger:

    def __init__(self, configs):
        self._configs = configs
        self.log_dir = configs['log_dir']
        self._writer = SummaryWriter(log_dir=self.log_dir)

        self.std_dev = None
        self.score = None
        self.scores = []  # list containing scores from each episode
        self.avg_score = None
        self.avg_scores = []  # list containing average scores after each episode
        self.scores_window = deque(maxlen=100)  # last 100 scores
        self.best_avg_score = -np.Inf  # best score for a single episode
        self.total_steps = 0  # track cumulative steps taken
        self.time_start = time.time()  # track cumulative wall time

    def update(self, steps, rewards, epi_nums):
        self.total_steps += steps
        self.score = sum(rewards)
        self.scores_window.append(self.score)
        self.scores.append(self.score)
        self.avg_score = np.mean(self.scores_window)
        self.avg_scores.append(self.avg_score)
        self.std_dev = np.std(self.scores_window)

        # update best average score
        if self.avg_score > self.best_avg_score and epi_nums > 100:
            self.best_avg_score = self.avg_score

    def print_epi(self, epi_num, steps, other_format, *args):
        stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f}  \
                        σ: {:8.3f}  |  Steps: {:8}   Reward: {:8.3f}  |  '.format(
            epi_num, self.avg_score, self.best_avg_score, self.std_dev, steps, self.score)

        print('\r' + stats + other_format.format(*args))
        self._writer.add_scalar('data/reward', self.score, epi_num)
        self._writer.add_scalar('data/std_dev', self.std_dev, epi_num)
        self._writer.add_scalar('data/avg_reward', self.avg_score, epi_num)

    def print_epoch(self, epi_num, other_format, *args):
        n_secs = int(time.time() - self.time_start)
        stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f} \
                σ: {:8.3f}  |  Steps: {:8}   Secs: {:6}      |  '.format(
            epi_num, self.avg_score, self.best_avg_score, self.std_dev, self.total_steps, n_secs)

        print('\r' + stats + other_format.format(*args))

    def print_solve(self, i_episode, stats_format, *args):
        self.print_epoch(i_episode, stats_format, *args)
        print('\nSolved in {:d} episodes!'.format(i_episode - 100))
        print('\nSolved in {:d} episodes!'.format(i_episode - 100), file=open(self.log_file_name, 'a'))

    def plot(self):
        plt.figure(1)
        subplot(211, self.scores, y_label='Score')
        subplot(212, self.avg_scores, y_label='Avg Score', x_label='Episodes')
        plt.show()

    def is_solved(self, epi_num, score_to_solve):
        return self.avg_score >= score_to_solve and epi_num > 100
