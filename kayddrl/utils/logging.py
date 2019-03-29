import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from colorama import Style, Fore
# from torch.utils.tensorboard import SummaryWriter

from kayddrl.utils.viz import subplot


class Logger:
    # Here will be the instance stored.
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Logger.__instance == None:
            Logger()
        return Logger.__instance

    def __init__(self, configs={'log_dir': 'runs'}):
        """ Virtually private constructor. """
        if Logger.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Logger.__instance = self
        self.time_origin = time.time()
        self.filename = None
        self.LEVELS = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3, 'FATAL': 4}
        self.COLORS = {'DEBUG': Style.DIM, 'INFO': Fore.GREEN, 'WARN': Fore.YELLOW, 'ERROR': Fore.RED,
                       'FATAL': Fore.RED + Style.BRIGHT}
        self.log_level = 1

        self._configs = configs
        # self._writer = SummaryWriter(log_dir=configs['log_dir'])

        self.std_dev = None
        self.score = None
        self.scores = []  # list containing scores from each episode
        self.avg_score = None
        self.avg_scores = []  # list containing average scores after each episode
        self.scores_window = deque(maxlen=100)  # last 100 scores
        self.best_avg_score = -np.Inf  # best score for a single episode
        self.total_steps = 0  # track cumulative steps taken
        self.start_time = time.time()  # track cumulative wall time

    def set_logfile(self, filename):
        self.filename = filename

    def set_log_level(self, log_level):
        self.log_level = log_level

    def _log(self, *args, sep=' ', end='\n', flush=False, level='INFO'):
        if self.LEVELS[level] >= self.log_level:
            elapsed_time = '[%12.5fs, %5s]' % (time.time() - self.time_origin, level)
            print(self.COLORS[level], elapsed_time, *args, Style.RESET_ALL, sep=sep, end=end, flush=flush)
            if self.filename:
                f = open(self.filename, 'a')
                print(elapsed_time, *args, sep=sep, end=end, flush=flush, file=f)
                f.close()

    def __call__(self, *args, sep=' ', end='\n', flush=False, level='INFO'):
        self._log(*args, sep=sep, end=end, flush=flush, level=level)

    def debug(self, *args, sep=' ', end='\n', flush=False):
        self._log(*args, sep=sep, end=end, flush=flush, level='DEBUG')

    def info(self, *args, sep=' ', end='\n', flush=False):
        self._log(*args, sep=sep, end=end, flush=flush, level='INFO')

    def warn(self, *args, sep=' ', end='\n', flush=False):
        self._log(*args, sep=sep, end=end, flush=flush, level='WARN')

    def error(self, *args, sep=' ', end='\n', flush=False):
        self._log(*args, sep=sep, end=end, flush=flush, level='ERROR')

    def fatal(self, *args, sep=' ', end='\n', flush=False):
        self._log(*args, sep=sep, end=end, flush=flush, level='FATAL')

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

        print('\r' + stats)  # + other_format.format(*args))
        self._writer.add_scalar('data/reward', self.score, epi_num)
        self._writer.add_scalar('data/std_dev', self.std_dev, epi_num)
        self._writer.add_scalar('data/avg_reward', self.avg_score, epi_num)

    def print_epoch(self, epi_num, other_format, *args):
        n_secs = int(time.time() - self.start_time)
        stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f} \
                σ: {:8.3f}  |  Steps: {:8}   Secs: {:6}      |  '.format(
            epi_num, self.avg_score, self.best_avg_score, self.std_dev, self.total_steps, n_secs)

        print('\r' + stats + other_format.format(*args))

    def print_solved(self, i_episode, stats_format, *args):
        self.print_epoch(i_episode, stats_format, *args)
        print('\nSolved in {:d} episodes!'.format(i_episode - 100))

    def plot(self):
        plt.figure(1)
        subplot(211, self.scores, y_label='Score')
        subplot(212, self.avg_scores, y_label='Avg Score', x_label='Episodes')
        plt.show()

    def is_solved(self, epi_num, score_to_solve):
        return self.avg_score >= score_to_solve and epi_num > 100


logger = Logger.getInstance()

if __name__ == '__main__':
    logger.set_log_level(0)
    logger.debug('This is a  debug message!')
    logger.info('This is an info  message!')
    logger.warn('This is a  warn  message!')
    logger.error('This is an error message!')
    logger.fatal('This is a  fatal message!')
