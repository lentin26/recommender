import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt


class Random:
    def __init__(self, n_users, n_items, time_buckets):
        # number of user, items
        self.n_users = n_users
        self.n_items = n_items
        # precision log
        self.cumulative_precision = []
        # reward logs
        self.rewards_log = [0] * time_buckets
        # update counter
        self.a = [1] * time_buckets
        # track user eligible items
        self.eligible_items = [np.copy(np.arange(n_items)) for _ in range(n_users)]
        # currently selected arm
        self.recommended_arm = None
        self.trace = []  # average reward trace
        self.arm_trace = []  # trace of arm recommendations

    def select_arm(self):
        """
        Make recommendations by randomly selecting among the options
        :return:
        """
        self.recommended_arm = choice(self.n_items)
        return self.recommended_arm

    def evaluate_policy(self, user_id, item_id, reward, t):
        """
      Trigger recommendation every period. Only count recomendation if
      the user was actually served the same recommendation. Record precision.
      Li et al. http://proceedings.mlr.press/v26/li12a/li12a.pdf
      """
        # select random arm
        arm = self.select_arm()
        # check for impression
        if arm == item_id:
            # update average reward log
            self.rewards_log[t] = self.rewards_log[t] + (1 / self.a[t]) * (reward - self.rewards_log[t])
            # update counter
            self.a[t] += 1

    def get_average_ctr(self):
        """
        Return average click-through-rate
        :return:
        """
        return self.rewards_log, self.a

    def plot_cumulative_precision(self):
        """
        Plot precision at each time step
        :return:
        """
        precision = np.array(self.cumulative_precision).reshape(self.n_users, self.T).mean(axis=0)
        plt.plot(np.arange(self.T), precision)
        plt.show()

    def get_results(self):
        """
        Get average reward and impression log
        :return: average rating at time t
        """
        return self.rewards_log, self.a

    def get_trace(self):
        """
        Return trace
        :return:
        """
        return self.trace, self.arm_trace