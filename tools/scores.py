''' Utility file for scores on predictions'''

import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import numpy as np

class Losses():
    """
      Point scores evaluations
    """

    def __init__(self, y_true, y_pred):
        '''Constructor of the class'''
        self.y_true = y_true
        self.y_pred = y_pred
        self.num_days = int(len(y_true) / 24)

    def mse_loss_avg(self):
        '''Method that computes overall Mean Squared Error (MSE)'''
        return tf.reduce_mean(tf.square(self.y_true - self.y_pred)).numpy()

    def rmse_loss_avg(self):
        '''Method that computes overall Root Mean Squared Error (MSE)'''
        return tf.sqrt(tf.reduce_mean(tf.square(self.y_true - self.y_pred))).numpy()

    def mae_loss_avg(self):
        '''Method that computes overall Mean Absolute Error (MAE)'''
        return tf.reduce_mean(tf.abs(self.y_true - self.y_pred)).numpy()

    def sMAPE_loss_avg(self):
        '''Method that computes overall Mean Absolute Percentage Error (MAPE)'''
        return tf.reduce_mean(
            tf.abs((self.y_true - self.y_pred)) / (0.5 * (tf.abs(self.y_true) + tf.abs(self.y_pred)))).numpy()

    def mse_loss_hourly(self):
        '''Method that computes Mean Squared Error (MSE) per hour'''
        y_true_reshaped = tf.reshape(self.y_true, (24, self.num_days))
        y_pred_reshaped = tf.reshape(self.y_pred, (24, self.num_days))

        # Calculate the squared differences
        squared_diff = tf.square(y_true_reshaped - y_pred_reshaped)

        # Calculate the mean squared error for each hour across all days
        mse_per_hour = tf.reduce_mean(squared_diff, axis=1)

        # Convert to a NumPy array if needed (for eager execution)
        return mse_per_hour.numpy()

    def sMAPE_loss_hourly(self):
        '''Method that computes Mean Absolute Percentage Error (MAPE) per hour'''
        y_true_reshaped = tf.reshape(self.y_true, (24, self.num_days))
        y_pred_reshaped = tf.reshape(self.y_pred, (24, self.num_days))

        sMAPE_per_hour = tf.reduce_mean(
            tf.abs(y_true_reshaped - y_pred_reshaped) / (0.5 * (tf.abs(y_true_reshaped) + tf.abs(y_pred_reshaped))),
            axis=1)

        return sMAPE_per_hour.numpy()

    def mae_loss_hourly(self):
        '''Method that computes Mean Absolute Error (MAE) per hour'''
        y_true_reshaped = tf.reshape(self.y_true, (24, self.num_days))
        y_pred_reshaped = tf.reshape(self.y_pred, (24, self.num_days))

        mae_per_hour = tf.reduce_mean(tf.abs(y_true_reshaped - y_pred_reshaped), axis=1)

        return mae_per_hour.numpy()

    def rmse_loss_hourly(self):
        '''Method that computes Root Mean Squared Error (RMSE) per hour'''
        y_true_reshaped = tf.reshape(self.y_true, (24, self.num_days))
        y_pred_reshaped = tf.reshape(self.y_pred, (24, self.num_days))

        # Calculate the squared differences
        squared_diff = tf.square(y_true_reshaped - y_pred_reshaped)

        # Calculate the mean squared error for each hour across all days
        mse_per_hour = tf.sqrt(tf.reduce_mean(squared_diff, axis=1))

        # Convert to a NumPy array if needed (for eager execution)
        return mse_per_hour.numpy()

    def mse_loss_daily(self):
        '''Method that computes Mean Squared Error (MSE) per day'''
        y_true_reshaped = tf.reshape(self.y_true, (24, self.num_days))
        y_pred_reshaped = tf.reshape(self.y_pred, (24, self.num_days))

        # Calculate the squared differences
        squared_diff = tf.square(y_true_reshaped - y_pred_reshaped)

        # Calculate the mean squared error for each day across all hours
        mse_per_hour = tf.reduce_mean(squared_diff, axis=0)

        # Convert to a NumPy array if needed (for eager execution)
        return mse_per_hour.numpy()

    def sMAPE_loss_daily(self):
        '''Method that computes Mean Absolute Percentage Error (MAPE) per day'''
        y_true_reshaped = tf.reshape(self.y_true, (24, self.num_days))
        y_pred_reshaped = tf.reshape(self.y_pred, (24, self.num_days))

        sMAPE_per_hour = tf.reduce_mean(
            tf.abs(y_true_reshaped - y_pred_reshaped) / (0.5 * (tf.abs(y_true_reshaped) + tf.abs(y_pred_reshaped))),
            axis=0)

        return sMAPE_per_hour.numpy()

    def mae_loss_daily(self):
        '''Method that computes Mean Absolute Error (MAE) per day'''
        y_true_reshaped = tf.reshape(self.y_true, (24, self.num_days))
        y_pred_reshaped = tf.reshape(self.y_pred, (24, self.num_days))

        mae_per_hour = tf.reduce_mean(tf.abs(y_true_reshaped - y_pred_reshaped), axis=0)

        return mae_per_hour.numpy()

    def rmse_loss_daily(self):
        '''Method that computes Root Mean Squared Error (RMSE) per day'''
        y_true_reshaped = tf.reshape(self.y_true, (24, self.num_days))
        y_pred_reshaped = tf.reshape(self.y_pred, (24, self.num_days))

        # Calculate the squared differences
        squared_diff = tf.square(y_true_reshaped - y_pred_reshaped)

        # Calculate the mean squared error for each day across all hours
        mse_per_hour = tf.sqrt(tf.reduce_mean(squared_diff, axis=0))

        # Convert to a NumPy array if needed (for eager execution)
        return mse_per_hour.numpy()


def compute_pinball_scores(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the pinball score on the test results
    return: pinball scores computed for each quantile level and each step in the pred horizon
    """
    score = []

    for i, q in enumerate(quantiles_levels):
        error = np.subtract(y_true, pred_quantiles[:, :, i])
        loss_q = np.maximum(q * error, (q - 1) * error)
        score.append(np.expand_dims(loss_q,-1))
    score = np.mean(np.concatenate(score, axis=-1), axis=0)
    return score

def compute_winkler_scores(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the winkler score on the test results
    return: winkler scores computed for each quantile level and each step in the pred horizon
    """
    score = []
    for i, q in enumerate(quantiles_levels[:len(quantiles_levels)//2+1]):
        L_n=pred_quantiles[:,:,i]
        U_n=pred_quantiles[:,:,-i-1]

        error=np.subtract(U_n,L_n)+2/(1-q)*(L_n-y_true)*(y_true<L_n)+2/(1-q)*(y_true-U_n)*(y_true>U_n)
        score.append(np.expand_dims(error,-1))

    score = np.mean(np.concatenate(score, axis=-1), axis=0)
    return score

def compute_delta_coverage(y_true, pred_quantiles, quantiles_levels, alpha_min, alpha_max, configs):
    """
    Calculate the Delta Coverage as defined based on the given lower bounds, upper bounds,
    actual values, and the range of alpha values. We assume the target alpha values have been defined
    in the range [alpha_min, alpha_max]
    """
    delta_coverage_sum = 0
    n=len(y_true)

    for i, q in enumerate(quantiles_levels[:len(quantiles_levels) // 2 ]):
        print(q)
        L_n = pred_quantiles[:, :, i]
        U_n = pred_quantiles[:, :, -i-1]
        hit_or_miss = (L_n <= y_true) & (y_true <= U_n)
        indicator_sum = np.sum(hit_or_miss)
        empirical_coverage = (indicator_sum / (n*configs['model_config']['pred_horiz'])) * 100
        delta_coverage_sum += abs(empirical_coverage - (1-2*q) * 100)

    delta_coverage_avg = delta_coverage_sum / (100*(alpha_max - alpha_min))
    print(f"Average Delta Coverage: {delta_coverage_avg:.5f}")
    return delta_coverage_avg

def compute_scores_totquant(score):
    """
    Utility function to compute the average score on the test results
    return: scores computed for each hour among all quantiles
    """
    return np.mean(score, axis=1)

def compute_scores_tottime(score):
    """
    Utility function to compute the average score on the test results
    return: scores computed for each quantile level in the total pred horizon
    """
    return np.mean(score, axis=0)

def compute_pinball_score_perday(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the average Pinball score on the test results
    return: scores computed for each day among all quantiles
    """

    score = []

    for i, q in enumerate(quantiles_levels):
        error = np.subtract(y_true, pred_quantiles[:, :, i])
        loss_q = np.maximum(q * error, (q - 1) * error)
        score.append(np.expand_dims(loss_q, -1))
    score_for_each_node=np.concatenate(score, axis=-1)
    scores_per_day=np.mean(score_for_each_node, axis=(1,2))
    return scores_per_day

def compute_winkler_score_perday(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the average Winkler score on the test results
    return: scores computed for each day among all quantiles
    """
    score = []

    score = []
    for i, q in enumerate(quantiles_levels[:len(quantiles_levels) // 2 + 1]):
        L_n = pred_quantiles[:, :, i]
        U_n = pred_quantiles[:, :, -i-1]

        error = np.subtract(U_n, L_n) + 2 / (1 - q) * (L_n - y_true) * (y_true < L_n) + 2 / (1 - q) * (y_true - U_n) * (
                    y_true > U_n)
        score.append(np.expand_dims(error, -1))


    score_for_each_node=np.concatenate(score, axis=-1)
    scores_per_day=np.mean(score_for_each_node, axis=(1,2))
    return scores_per_day


def compute_avg_scores_(score):
    """
    Utility function to compute the average score on the test results
    return: pinball scores computes the mean of the  score
    """
    L = tf.convert_to_tensor(score)
    return tf.reduce_mean(L).numpy()

def plot_scores(score: np.ndarray, score_type: str, configs: dict, ensemble: str):
    """
    Utility function to plot the scores in two cases:
     1. Having aggregated the scores in quantiles per hour
     2. Having aggregated the scores in quantiles per day
     """

    if score.ndim == 1:
        if ensemble == "Score_per_hour":
            plt.bar(range(configs['model_config']['pred_horiz']), score)
            plt.xticks(range(configs['model_config']['pred_horiz']), ['Hour {}'.format(i) for i in range(configs['model_config']['pred_horiz'])])
            plt.xticks(rotation=45)
            plt.ylabel(score_type)
            plt.title('Scores for Each Hour')
            plt.show()

        elif ensemble=="Score_per_day":
            plt.bar(range(len(score)), score)
            plt.xticks(range(0, len(score), 30), ['Day {}'.format(i) for i in range(1, len(score) + 1, 30)])

            plt.ylabel(score_type)
            plt.title('Scores for Each Day')
            plt.show()

        else:
            sys.exit("Error: choose a valid score ensemble")

