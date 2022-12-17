from BayesMatchBase import BayesMatchBase
from scipy.stats import multinomial
from numpy import stack, concatenate
import pandas as pd


class BayesMatch(BayesMatchBase):

    def __init__(self, u_data, v_data, n_common, burn_in, n_samples, eval_every, n_clusters, *args, **kwargs):
        """

        :param n_common: number of common features
        :param u_data: users data with first n_common columns corresponding to common features
        :param v_data: item data with first n_common columns corresponding to common features
        :param args:
        :param kwargs:
        """
        super().__init__(u_data, v_data, n_common, n_clusters)

        # convert to numpy
        if isinstance(u_data, pd.DataFrame):
            u_data = u_data.to_numpy()
        if isinstance(v_data, pd.DataFrame):
            v_data = v_data.to_numpy()

        self.u_data = u_data
        self.v_data = v_data
        self.n_common = n_common
        self.n_clusters = n_clusters

        self.burn_in = burn_in
        self.n_samples = n_samples
        self.eval_every = eval_every
        self.iteration = 0

    def random_init(self):
        # randomly initialize user data
        n = self.n_common
        n_cl = self.n_clusters
        n_users = len(self.u_data)
        n_items = len(self.v_data)
        # user cluster assignment
        self.ufa = multinomial(1, n_cl * [1 / n_cl]).rvs(n_users).argmax(axis=1)
        # item cluster assignment
        self.ifa = multinomial(1, n_cl * [1 / n_cl]).rvs(n_items).argmax(axis=1)
        for view, data in enumerate([self.u_data, self.v_data]):
            if view == 0:
                ca = self.ufa
            elif view == 1:
                ca = self.ifa
            else:
                raise Exception("Selected view must either be 0 or 1.")
            for uid, fv in enumerate(data):
                # split user feature values into common and user-specific
                c, v = fv[:n], fv[n:]
                k = int(ca[uid])
                self.update(uid, k, v=v, c=c, view=view, i=1)

    def _sample(self, view):
        """
        Samples new cluster assignments for features in all members and updates the current state of the posterior
        """

        n = self.n_common
        # get data
        if view == 0:
            data, fa = self.u_data, self.ufa
            cf = self.ucf
            phi_trace = self.phi_u_trace
        elif view == 1:
            data = self.v_data
            fa = self.ifa
            cf = self.icf
            phi_trace = self.phi_v_trace
        else:
            raise Exception("Selected view must either be 0 or 1.")
        for uid, fv in enumerate(data):
            if uid % 100 == 0:
                print("{} / {}\t\t".format(uid, len(data)), end="\r")
            # split user feature values into common and user-specific
            c, v = fv[:n], fv[n:]
            # get current cluster assignment
            k = int(fa[uid])
            # decrement all corpus statistics by one
            self.update(uid, k, v, c, view, i=-1)
            # compute full conditional posterior vector
            probs = self.get_conditional_prob(uid, v, c, view)
            # sample new cluster_idx, returns index of vector of all 0s and one 1
            k_new = multinomial(1, probs).rvs().argmax()
            # increment all corpus statistics by on
            self.update(uid, k_new, v, c, view, i=1)
            fa[uid] = k_new

            # update parameter traces
            self.u_theta_trace.append(self.ucc.sum(axis=0))
            self.v_theta_trace.append(self.icc.sum(axis=0))
            for x in range(len(v)):
                phi_trace[x].append(cf[x + len(c)].sum(axis=0))
            for x in range(len(c)):
                phi_trace[x].append(cf[x].sum(axis=0))
            # self.phi_c_trace = [[self.ccf[x][:, nu] for x, nu in enumerate(c)]]

    def fit(self, optimize_priors=False):
        """
        Fits BayesMatch model using collapsed Gibbs sampling
        :return:
        """
        self.random_init()
        for iteration in range(self.burn_in + self.n_samples):
            # 1) generate user and item samples from the chain
            self._sample(view=0)  # sample user
            self._sample(view=1)  # sample item

            # trace metrics to ensure convergence
            self.trace_metrics()
            # print log likelihood and perplexity
            log_likelihood = self.log_likelihood_trace[-1]
            perplexity = self.perplexity_trace[-1]
            if iteration >= self.burn_in:
                print("sampling iteration %i perplexity %.1f likelihood %.1f" % (
                    iteration, round(perplexity, 2), round(log_likelihood, 1)), end="\r")

                # early stopping
                # if perplexity - ([0] + self.perplexity_trace)[-2] <= 10e-10:  # add padding
                #     break
            else:
                print("burn-in iteration %i perplexity %.1f likelihood %.1f" % (
                    iteration, round(perplexity, 2), round(log_likelihood, 1)), end="\r")


if __name__ == "__main__":
    import pandas as pd

    # book recommender
    user = pd.read_csv('data/books/users_enc.csv')
    item = pd.read_csv('data/books/books_enc.csv')

    # instantiate BayesMatch
    m = 1000
    n_clusters = 2
    match = BayesMatch(
        u_data=user.drop('user_id', axis=1).head(m),
        v_data=item.drop('item_id', axis=1).head(m),
        n_common=0,
        n_clusters=n_clusters,
        burn_in=0,
        n_samples=10,
        eval_every=100
    )

    # fit data
    match.fit()

    import matplotlib.pyplot as plt
    # plot results
    burn_in = match.burn_in
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(match.log_likelihood_trace[burn_in:])
    ax[1].plot(match.perplexity_trace[burn_in:])

    plt.suptitle('Book Recommender Clustering')
    ax[0].set_ylabel('Log Likelihood')
    ax[1].set_ylabel('Perplexity')
    plt.show()


