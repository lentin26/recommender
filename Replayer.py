import numpy as np


class Replayer:
    """Replayer method of evaluation"""
    def __init__(self, model, time_buckets):
        """
        ;param: model should have evaluate_policy method
        """
        self.model = model
        self.time_buckets = time_buckets

    def replay(self, ratings):
        """
        Run experiment on dataset using replayer method
        :param ratings: dataset [user_id, item_id, rating and time bucket]
        :return:
        """
        # run experiment
        i = 0
        for user_id, item_id, rating, t in ratings.to_numpy():
            self.model.evaluate_policy(user_id=int(user_id), item_id=int(item_id), reward=rating, t=int(t))
            results, impressions = self.get_results()
            print(
                "Progress", round(i / len(ratings), 3),
                'Time Bucket:', int(t),
                "Impressions:", impressions[int(t)] - 1,
                "User ID:", int(user_id),
                "Historical Arm:", item_id,
                "Selected Arm:", self.model.recommended_arm,
                "Average Rating:", results[int(t)], end="\r")
            i += 1

    def get_results(self):
        """
        Get average reward and impression log
        :return: average rating at time t
        """
        return self.model.rewards_log, self.model.a

    def get_trace(self):
        """
        Return trace
        :return:
        """
        return self.model.trace, self.model.arm_trace


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from datetime import date

    # get ratings data
    ratings = pd.read_csv('data/ml-100k/ratings_enc.csv')
    ratings = ratings[['user_id', 'item_id', 'rating', 'time_bucket']]

    # get model params from data
    T = ratings.time_bucket.nunique()  # get number of unique time buckets
    T = 5
    # get number of users, items
    n_users = ratings.user_id.nunique()  # number of users
    n_items = ratings.item_id.nunique()  # number of items

    from ICTR import ICTR

    # instantiate model instance
    from Random import Random
    model = Random(
        n_users=n_users,
        n_items=n_items,
        time_buckets=1)

    # run experiment
    replayer = Replayer(model, time_buckets=T)
    replayer.replay(ratings)

    # get results
    avg_rating, impressions = model.get_results()
    reward_trace, arm_trace = model.get_trace()