from numpy import zeros, ones, exp, log, unique, concatenate, array
from scipy.special import loggamma
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from math import lgamma
import matplotlib.pyplot as plt


def encode_data(data):
    # encode
    encoder = LabelEncoder()
    for col in data.columns:
        data['{}'.format(col)] = encoder.fit_transform(data['{}'.format(col)])
        # print(user['{}'.format(col)].nunique() == user['{}'.format(col)].max() + 1)
    return data


class BayesMatchBase:

    def __init__(self, u_data, v_data, n_common, n_clusters):

        u_data = encode_data(u_data)
        v_data = encode_data(v_data)

        # convert to numpy
        if isinstance(u_data, pd.DataFrame):
            u_data = u_data.to_numpy()
        if isinstance(v_data, pd.DataFrame):
            v_data = v_data.to_numpy()

        self.u_data = u_data
        self.v_data = v_data

        # compute observed ontology sizes for user, item and common view features.
        # features are indicated by position, vocabulary/ontology size by value
        v_users = [len(unique(u_data[:, i])) for i in range(u_data.shape[1])]
        v_items = [len(unique(v_data[:, i])) for i in range(v_data.shape[1])]
        data = concatenate([u_data[:, :n_common], v_data[:, :n_common]]).copy()
        if data:
            v_common = [len(unique(data[:, i])) for i in range(data.shape[1])]
        else:
            v_common = []

        self.n = u_data.size + v_data.size

        n_users = len(u_data)
        n_items = len(v_data)
        self.v_users = v_users
        self.v_items = v_items
        # self.v_common = v_common
        self.n_clusters = n_clusters
        self.n_common = n_common

        # user cluster count
        self.ucc = zeros((n_users, n_clusters))
        # item cluster count
        self.icc = zeros((n_items, n_clusters))

        # user cluster assignment
        self.ufa = zeros(n_users)
        # item cluster assignment
        self.ifa = zeros(n_items)

        # user, cluster, user-specific vocab tensor for each feature
        self.ucf = [zeros((n_users, n_clusters, v_users[i])) for i in range(u_data.shape[1])]
        # item, cluster, item-specific vocab for each feature
        self.icf = [zeros((n_items, n_clusters, v_items[i])) for i in range(v_data.shape[1])]

        # dirichlet smoothing parameters
        self.alpha = 100
        self.beta = 100

        # parameters to learn
        # self.theta_u = zeros(n_clusters)
        # self.theta_v = zeros(n_clusters)
        # self.phi_u = [zeros((n_clusters, v_users[i])) for i in range(len(v_users))]
        # self.phi_v = [zeros((n_clusters, v_items[i])) for i in range(len(v_items))]
        # self.phi_c = [zeros((n_clusters, v_common[i])) for i in range(len(v_common))]

        # likelihood, perplexity trace
        self.log_likelihood_trace = []
        self.perplexity_trace = []
        self.u_theta_trace = []
        self.v_theta_trace = []
        self.phi_u_trace = [[] for _ in range(len(v_users))]
        self.phi_v_trace = [[] for _ in range(len(v_items))]
        self.phi_c_trace = [[] for _ in range(len(v_common))]
        # self.normalizing_const = self._get_const()

    def update(self, uid, k, v, c, view, i):
        """

        :param uid: user-id
        :param v: array-like containing value indices for features 1, 2, ... , F^{(u/v)}
        :param c: array-like containing value indices for features 1, 2, ... , F^{(c)}
        :param k: cluster index
        :param i: +1 or -1
        :param view: 0 or 1 corresponding to user or item view, respectively
        :return:
        """
        # select a view
        if view == 0:
            cf, cc = self.ucf, self.ucc
        elif view == 1:
            cf, cc = self.icf, self.icc
        else:
            raise Exception("Selected view must either be 0 or 1.")

        # decrement cluster-user feature count
        if i == -1:
            for j, x in enumerate(v):
                # v += len(c)
                cf[j][uid, k, x] += i
            # decrement cluster-common feature count
            for j, x in enumerate(c):
                cf[j][uid, k, x] += i
        if i == 1:
            for j, x in enumerate(v):
                # v += len(c)
                cf[j][:, k, x] += i
                # decrement cluster-common feature count
            for j, x in enumerate(c):
                cf[j][:, k, x] += i
        # increment/decrement user cluster count
        cc[uid, k] += i

    def get_conditional_prob(self, uid, v, c, view):

        alpha = self.alpha
        beta = self.beta
        # select a view
        if view == 0:
            # get user cluster count
            cc, cf, vocab_sizes = self.ucc.sum(axis=0) + alpha, self.ucf, self.v_users
        elif view == 1:
            # get user cluster count
            cc, cf, vocab_sizes = self.icc.sum(axis=0) + alpha, self.icf, self.v_items
        else:
            raise Exception("Selected view must either be 0 or 1.")

        # get user-feature counts
        user_prod = 1
        for i, x in enumerate(v):
            user_prod *= cf[i][:, :, x].sum(axis=0) + beta
            user_prod /= cf[i][:, :, :].sum(axis=0).sum(axis=1) + vocab_sizes[i] * beta

        # get common-feature counts
        common_prod = 1
        for i, x in enumerate(c):
            common_prod *= cf[i][:, :, x].sum(axis=0) + beta
            common_prod /= cf[i][:, :, :].sum(axis=0).sum(axis=1) + vocab_sizes[i] * beta

        # full conditional probability
        cond_prob = cc * user_prod * common_prod
        return cond_prob / sum(cond_prob)

    def _loglikelihood(self):
        beta = self.beta
        ll = 0
        n = 0
        for view, data in enumerate([self.u_data, self.v_data]):
            if view == 0:
                cc, cf, vocab_sizes = self.ucc, self.ucf, self.v_users
            elif view == 1:
                cc, cf, vocab_sizes = self.icc, self.icf, self.v_items
            else:
                raise Exception("Selected view must either be 0 or 1.")
            for uid, fv in enumerate(data):
                cs, vs = fv[:n], fv[n:]
                for x, v in enumerate(vs):
                    # number of feature = v assignments
                    ll += log(sum(cf[x][:, :, v].sum(axis=0) + beta))
                    # number of feature assignments
                    ll -= log(sum(cf[x][:, :, :].sum(axis=0).sum(axis=1) + vocab_sizes[x] * beta))
                for x, v in enumerate(cs):
                    # number of feature = v assignments
                    ll += log(sum(cc[x][:, :,  v] + beta)).sum(axis=0)
                    # number of feature assignments
                    ll -= log(sum(cc[x][:, :, :].sum(axis=0) .sum(axis=1) + vocab_sizes[x] * beta))
        return ll

    def trace_metrics(self):
        """
        Traces metrics to ensure convergence
        """
        # get log likelihood
        log_likelihood = self._loglikelihood()
        # number of data points
        n = self.n
        # compute perplexity
        perplexity = 2 ** (-log_likelihood / n)
        self.log_likelihood_trace.append(log_likelihood)
        self.perplexity_trace.append(perplexity)




