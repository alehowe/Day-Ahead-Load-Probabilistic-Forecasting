'''
Functions for conformal prediction
'''


import sys
from typing import List
import numpy as np
import os
import pandas as pd
import subprocess
from typing import Dict



def build_target_quantiles(target_alpha: List):
    """
    Build target quantiles from the list of alpha, including the median
    """
    target_quantiles = [0.5]
    for alpha in target_alpha:
        target_quantiles.append(alpha / 2)
        target_quantiles.append(1 - alpha / 2)
    target_quantiles.sort()
    return target_quantiles


def build_cp_pis(preds_cali: np.array, y_cali: np.array, preds_test: np.array,
                 settings: Dict, method: str='higher'):
    """
    Compute PIs at the different alpha levels using conformal prediction
    """
    #Squeeze the recalibration predictions
    preds_cali = np.squeeze(preds_cali, axis=-1)
    if preds_test.shape[0]>1:
        sys.exit('ERROR: exec_cup supports single test samples')
    # Compute conformity score (absolute residual)
    conf_score = np.abs(preds_cali - y_cali)
    #Evaluate the number of available scores
    n=conf_score.shape[0]
    preds_test_q=[preds_test]
    # Stack the quantiles to the point pred for each alpha
    for alpha in settings['target_alpha']:
        # Compute the quantile for the current alpha
        q = np.ceil((n + 1) * (1 - alpha)) / n
        #Adapt the dimension of the quantile
        Q_1_alpha= np.expand_dims(np.quantile(a=conf_score, q=q, axis=0, method=method),
                                  axis=(0,-1))
        # Append lower/upper PIs for the current alpha
        preds_test_q.append(preds_test - Q_1_alpha)
        preds_test_q.append(preds_test + Q_1_alpha)
    # Concatenate the PIs for each alpha
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    # Fix quantile crossing by sorting and return prediction flattened in temporal dimension (sample over pred horizon)
    return np.sort(preds_test_q.reshape(-1, preds_test_q.shape[-1]), axis=-1)

def compute_cp(recalib_preds, settings: Dict):
    """
    Reshape recalibration predictions and execute conformal prediciton for each test sample
    """
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    ens_p = recalib_preds.loc[:,0.5].to_numpy()
    ens_p_d = ens_p.reshape(-1, settings['pred_horiz'], 1)
    #Extract the target data
    target_d = recalib_preds.filter([settings['task_name']], axis=1).to_numpy().reshape(-1, settings['pred_horiz'])
    #Evaluate the number of remaining test samples
    num_test_samples = ens_p_d.shape[0] - settings['num_cali_samples']
    #Create an empty list to store the PIs for each test sample
    test_PIs=[]
    for t_s in range(num_test_samples):
        #Extract settings['num_cali_samples'] recalibration predictions
        preds_cali = ens_p_d[t_s:settings['num_cali_samples'] + t_s]
        #Extract the test prediction for the current test sample
        preds_test = ens_p_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s+1]
        # Extract settings['num_cali_samples'] target values
        y_cali = target_d[t_s:settings['num_cali_samples'] + t_s]
        # Apply conformal prediction to compute PIs
        test_PIs.append(build_cp_pis(preds_cali=preds_cali,
                                     y_cali=y_cali,
                                     preds_test=preds_test,
                                     settings=settings))
    # Concatenate the PIs for each test sample
    test_PIs=np.concatenate(test_PIs, axis=0)
    # Build updated dataframe
    aggr_df=recalib_preds.filter([settings['task_name']], axis=1)
    aggr_df=aggr_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]
    for j in range(len(settings['target_quantiles'])):
        aggr_df[settings['target_quantiles'][j]]=test_PIs[:,j]
    return aggr_df

def build_cqr_pis(preds_cali: np.array, preds_cali_low: np.array, preds_cali_high: np.array,
                  y_cali: np.array, preds_test: np.array, preds_test_low: np.array, preds_test_high: np.array,
                  settings: Dict, method: str = 'higher'):
    """
    Compute PIs at the different alpha levels using conformalized quantile regression
    """
    # Adapt to the dimension of the recalibration predictions
    y_cali = np.expand_dims(y_cali, axis=-1)
    if preds_test_low.shape[0] > 1 or preds_test_high.shape[0] > 1:
        sys.exit('ERROR: exec_cqr supports single test samples')
    # Compute conformity score (absolute residual)
    conformity_scores = np.maximum(preds_cali_low - y_cali, y_cali - preds_cali_high)
    # Evaluate the number of available scores
    n = conformity_scores.shape[0]
    preds_test_q = [preds_test]
    # Stack the quantiles to the point pred for each alpha
    for alpha_idx, alpha in enumerate(settings['target_alpha']):
        # Compute the quantile for the current alpha
        q = np.ceil((n + 1) * (1 - alpha)) / n
        # Adapt the dimension of the quantile
        Q_1_alpha = np.quantile(a=np.expand_dims(conformity_scores[:, :, alpha_idx], axis=-1), q=q, axis=0,
                                method=method)
        # Append lower/upper PIs for the current alpha
        preds_test_q.append(np.expand_dims(preds_test_low[:, :, alpha_idx] - np.transpose(Q_1_alpha), axis=-1))
        preds_test_q.append(np.expand_dims(preds_test_high[:, :, alpha_idx] + np.transpose(Q_1_alpha), axis=-1))
    # Concatenate the PIs for each alpha
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    # Fix quantile crossing by sorting and return prediction flattened in temporal dimension (sample over pred horizon)
    return np.sort(preds_test_q.reshape(-1, preds_test_q.shape[-1]), axis=-1)


def compute_cqr(recalib_preds, settings: Dict):
    """
    Reshape recalibration predictions and execute conformalized quantile regression for each test sample
    """
    #Reshape recalibration predictions
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    target_quantiles = settings['target_quantiles']
    #Select the lower target quantiles
    low_quantiles = [q for q in target_quantiles if q < 0.5]
    #Select the higher target quantiles
    high_quantiles = [q for q in target_quantiles if q > 0.5]
    #Reverse the order of the higher quantiles
    high_quantiles = high_quantiles[::-1]
    #Extract recalibration predictions
    ens_p = recalib_preds.loc[:, 0.5].to_numpy()
    #Extract recalibration predictions for the lower quantiles
    ens_p_low = recalib_preds.loc[:, low_quantiles].to_numpy()
    #Extract recalibration predictions for the higher quantiles
    ens_p_high = recalib_preds.loc[:, high_quantiles].to_numpy()
    #Adapt the dimension of the recalibration predictions
    ens_p_d = ens_p.reshape(-1, settings['pred_horiz'], 1)
    ens_p_low_d = ens_p_low.reshape(-1, settings['pred_horiz'], len(low_quantiles))
    ens_p_high_d = ens_p_high.reshape(-1, settings['pred_horiz'], len(high_quantiles))
    #Extract the target data
    target_d = recalib_preds.filter([settings['task_name']], axis=1).to_numpy().reshape(-1, settings['pred_horiz'])
    #Evaluate the number of remaining test samples
    num_test_samples = ens_p_d.shape[0] - settings['num_cali_samples']
    test_PIs = []
    #Apply conformalized quantile regression for each test sample
    for t_s in range(num_test_samples):
        #Extract settings['num_cali_samples'] recalibration predictions
        preds_cali = ens_p_d[t_s:settings['num_cali_samples'] + t_s]
        #Extract settings['num_cali_samples'] recalibration predictions for the lower quantiles
        preds_cali_low = ens_p_low_d[t_s:settings['num_cali_samples'] + t_s]
        #Extract settings['num_cali_samples'] recalibration predictions for the higher quantiles
        preds_cali_high = ens_p_high_d[t_s:settings['num_cali_samples'] + t_s]
        #Extract the test prediction for the current test sample
        preds_test = ens_p_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s + 1]
        preds_test_low = ens_p_low_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s + 1]
        preds_test_high = ens_p_high_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s + 1]
        #Extract settings['num_cali_samples'] target values
        y_cali = target_d[t_s:settings['num_cali_samples'] + t_s]
        #Apply conformalized quantile regression to compute PIs
        test_PIs.append(build_cqr_pis(preds_cali=preds_cali,
                                      preds_cali_low=preds_cali_low,
                                      preds_cali_high=preds_cali_high,
                                      y_cali=y_cali,
                                      preds_test=preds_test,
                                      preds_test_low=preds_test_low,
                                      preds_test_high=preds_test_high,
                                      settings=settings))
    #Concatenate the PIs for each test sample
    test_PIs = np.concatenate(test_PIs, axis=0)
    # Build updated dataframe
    aggr_df = recalib_preds.filter([settings['task_name']], axis=1)
    aggr_df = aggr_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]
    for j in range(len(settings['target_quantiles'])):
        aggr_df[settings['target_quantiles'][j]] = test_PIs[:, j]
    return aggr_df


def build_aci_pis(preds_cali: np.array, y_cali: np.array, y_actual: np.array, preds_test: np.array,
                  settings: Dict, method: str = 'higher'):
    """
    Compute PIs at the different alpha levels using adaptive conformal inference
    """
    # Squeeze the recalibration predictions
    preds_cali = np.squeeze(preds_cali, axis=-1)
    if preds_test.shape[0] > 1:
        sys.exit('ERROR: exec_cup supports single test samples')
    # Compute conformity score (absolute residual)
    conf_score = np.abs(preds_cali - y_cali)
    # Evaluate the number of available scores
    n = conf_score.shape[0]
    preds_test_q = [preds_test]
    # Stack the quantiles to the point prediction for each alpha
    for alphaidx, alpha in enumerate(settings['target_alpha']):
        # Compute the quantile for the current alpha
        q = np.ceil((n + 1) * (1 - alpha)) / n
        # Adapt the dimension of the quantile
        Q_1_alpha = np.expand_dims(np.quantile(a=conf_score, q=q, axis=0, method=method), axis=(0, -1))
        # Append lower/upper PIs for the current alpha
        preds_test_q.append(preds_test - Q_1_alpha)
        preds_test_q.append(preds_test + Q_1_alpha)
        # Check if the proportion of hourly target values outside the PI is >= the original alpha
        if np.sum(np.logical_or(y_actual.reshape(-1, 1) >= np.squeeze((preds_test + Q_1_alpha), axis=0),
                                y_actual.reshape(-1, 1) <= np.squeeze((preds_test - Q_1_alpha), axis=0))) / settings[
            'pred_horiz'] >= settings['original_alpha'][alphaidx]:
            #substitute the new alpha value
            settings['target_alpha'][alphaidx] = settings['target_alpha'][alphaidx] + settings['lr'] * (
                        settings['original_alpha'][alphaidx] - 1)
        else:
            settings['target_alpha'][alphaidx] = settings['target_alpha'][alphaidx] + settings['lr'] * \
                                                 settings['original_alpha'][alphaidx]
    # Concatenate the PIs for each alpha
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    # Fix quantile crossing by sorting and return prediction flattened in temporal dimension (sample over pred horizon)
    return np.sort(preds_test_q.reshape(-1, preds_test_q.shape[-1]), axis=-1)


def compute_aci(recalib_preds, settings: Dict):
    """
    Reshape recalibration predictions and execute adaptive conformal inference for each test sample
    """
    # Reshape recalibration predictions
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    ens_p = recalib_preds.loc[:, 0.5].to_numpy()
    ens_p_d = ens_p.reshape(-1, settings['pred_horiz'], 1)
    # Extract the target data
    target_d = recalib_preds.filter([settings['task_name']], axis=1).to_numpy().reshape(-1, settings['pred_horiz'])
    # Evaluate the number of remaining test samples
    num_test_samples = ens_p_d.shape[0] - settings['num_cali_samples']
    test_PIs = []
    # Save the original alpha values
    settings['original_alpha'] = settings['target_alpha']
    # Apply adaptive conformal inference for each test sample
    for t_s in range(num_test_samples):
        # Extract settings['num_cali_samples'] recalibration predictions
        preds_cali = ens_p_d[t_s:settings['num_cali_samples'] + t_s]
        # Extract the test prediction for the current test sample
        preds_test = ens_p_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s + 1]
        # Extract settings['num_cali_samples'] target values
        y_cali = target_d[t_s:settings['num_cali_samples'] + t_s]
        # Extract the target value for the current test sample
        y_actual = target_d[settings['num_cali_samples'] + t_s]
        # Apply adaptive conformal inference to compute PIs
        test_PIs.append(build_aci_pis(preds_cali=preds_cali,
                                      y_actual=y_actual,
                                      y_cali=y_cali,
                                      preds_test=preds_test,
                                      settings=settings))
    # Concatenate the PIs for each test sample
    test_PIs = np.concatenate(test_PIs, axis=0)
    # Build updated dataframe
    aggr_df = recalib_preds.filter([settings['task_name']], axis=1)
    aggr_df = aggr_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]
    for j in range(len(settings['target_quantiles'])):
        aggr_df[settings['target_quantiles'][j]] = test_PIs[:, j]
    return aggr_df


def build_naci_pis(preds_cali: np.array, y_cali: np.array, y_actual: np.array, preds_test: np.array, gamma_best_k: int,
                  tsample: int, gamma_values: np.array,
                  alphagamma: np.array, interval_lengths: np.array, valid_intervals: np.array, settings: Dict,
                  method: str = 'higher'):
    """
    Compute PIs at the different alpha levels using naive algorithm based on asymptotic conformal inference
    """
    # Squeeze the recalibration predictions
    preds_cali = np.squeeze(preds_cali, axis=-1)
    if preds_test.shape[0] > 1:
        sys.exit('ERROR: exec_cup supports single test samples')
    # Compute conformity score (absolute residual)
    conf_score = np.abs(preds_cali - y_cali)
    # Evaluate the number of available scores
    n = conf_score.shape[0]
    preds_test_q = [preds_test]
    # Stack the quantiles to the point prediction for each alpha and each gamma
    for gammaidx, gamma in enumerate(gamma_values):
        for alphaidx, alpha in enumerate(alphagamma[:, gammaidx]):
            # Compute the quantile for the current alpha
            q = np.ceil((n + 1) * (1 - alpha)) / n
            # Adapt the dimension of the quantile
            Q_1_alpha = np.expand_dims(np.quantile(a=conf_score, q=q, axis=0, method=method), axis=(0, -1))
            # Check if for this cycle the gamma is that corresponding to the value obtained for each alpha at previous test sample
            if gamma == gamma_best_k[alphaidx]:
                # Append lower/upper PIs for the current alpha
                preds_test_q.append(preds_test - Q_1_alpha)
                preds_test_q.append(preds_test + Q_1_alpha)
            # Check if the proportion of hourly target values outside the PI is >= the original alpha
            if np.sum(np.logical_or(y_actual.reshape(-1, 1) >= np.squeeze((preds_test + Q_1_alpha), axis=0),
                                    y_actual.reshape(-1, 1) <= np.squeeze((preds_test - Q_1_alpha), axis=0))) / \
                    settings['pred_horiz'] >= settings['original_alpha'][alphaidx]:
                # Update the alpha value
                alphagamma[alphaidx, gammaidx] += gamma * (settings['original_alpha'][alphaidx] - 1)
            else:
                alphagamma[alphaidx, gammaidx] += gamma * settings['original_alpha'][alphaidx]
            # Compute the interval length
            interval_lengths[alphaidx, tsample, gammaidx] = np.mean(
                np.squeeze(preds_test + Q_1_alpha, axis=0) - np.squeeze(preds_test - Q_1_alpha, axis=0))
            # Check if the proportion of hourly target values outside the PI is <= the original alpha
            valid_intervals[alphaidx, tsample, gammaidx] = np.sum(
                np.logical_or(y_actual.reshape(-1, 1) >= np.squeeze((preds_test + Q_1_alpha), axis=0),
                              y_actual.reshape(-1, 1) <= np.squeeze((preds_test - Q_1_alpha), axis=0))) / settings[
                                                               'pred_horiz'] <= settings['original_alpha'][alphaidx]
    # Concatenate the PIs for each alpha and gamma
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    # Fix quantile crossing by sorting and return prediction flattened in temporal dimension (sample over pred horizon)
    return np.sort(preds_test_q.reshape(-1, preds_test_q.shape[-1]), axis=-1)


def compute_naci(recalib_preds, settings: Dict):
    """
    Reshape recalibration predictions and execute naive algorithm based on asymptotic conformal inference for each test sample
    """
    # Reshape recalibration predictions
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    ens_p = recalib_preds.loc[:, 0.5].to_numpy()
    ens_p_d = ens_p.reshape(-1, settings['pred_horiz'], 1)
    # Extract the target data
    target_d = recalib_preds.filter([settings['task_name']], axis=1).to_numpy().reshape(-1, settings['pred_horiz'])
    # Evaluate the number of remaining test samples
    num_test_samples = ens_p_d.shape[0] - settings['num_cali_samples']
    test_PIs = []
    # Save the original alpha values
    settings['original_alpha'] = settings['target_alpha']
    # Save the gamma values
    gamma_values = settings['lr']
    # Save the number of gamma values
    K = len(gamma_values)
    # Save the length of the warm-up period
    Tw = settings['warm_up_period']
    # Initialize the matrix of alpha values for different gamma
    alphagamma = np.zeros((len(settings['target_alpha']), K))
    for k in range(K):
        alphagamma[:, k] = settings['target_alpha']
    # Initialize the tensor of lower and upper bounds
    interval_lengths = np.zeros((len(settings['target_alpha']), num_test_samples, K))
    # Initialize the tensor of valid intervals
    valid_intervals = np.zeros((len(settings['target_alpha']), num_test_samples, K))
    # Initialize the vector of best gamma values for each alpha
    gamma_best_k = np.zeros(len(settings['target_alpha']))
    # Apply naive algorithm based on asymptotic conformal inference for each test sample
    for t_s in range(num_test_samples):
        # Extract settings['num_cali_samples'] recalibration predictions
        preds_cali = ens_p_d[t_s:settings['num_cali_samples'] + t_s]
        # Extract the test prediction for the current test sample
        preds_test = ens_p_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s + 1]
        # Extract settings['num_cali_samples'] target values
        y_cali = target_d[t_s:settings['num_cali_samples'] + t_s]
        # Extract the target value for the current test sample
        y_actual = target_d[settings['num_cali_samples'] + t_s]
        # Apply naive algorithm based on asymptotic conformal inference to compute PIs
        if t_s < Tw:
            #for the first T_w test samples the best gamma is always the first one
            gamma_best_k[:] = gamma_values[0]
            # Print the vector

            # Apply naive algorithm based on asymptotic conformal inference to compute PIs
            test_PIs.append(build_naci_pis(preds_cali=preds_cali, y_actual=y_actual, y_cali=y_cali,
                                          preds_test=preds_test, gamma_best_k=gamma_best_k,
                                          tsample=t_s, gamma_values=gamma_values, alphagamma=alphagamma,
                                          interval_lengths=interval_lengths, valid_intervals=valid_intervals,
                                          settings=settings))


        else:
            # Evaluate the best gamma value for each alpha based on the results of the previous test_sample
            for alphaidx, alpha in enumerate(settings['original_alpha']):
                # Define the set of gamma values for which the proportion of valid intervals are greater than or equal to the original alpha
                A_t = [k for k in range(K) if np.mean(valid_intervals[alphaidx, :(t_s), k]) >= (1 - alpha)]

                # In case of non-empty set
                if A_t:
                    # Compute the best gamma value for the current alpha checking for all the previous test samples
                    minval = np.min(
                        np.abs(np.mean(interval_lengths[alphaidx, :(t_s), A_t], axis=1)))
                    best_k = np.max(np.where(np.abs(np.mean(interval_lengths[alphaidx, :(t_s), A_t], axis=1)) == minval))
                    # Update the best gamma value for the current alpha
                    gamma_best_k[alphaidx] = gamma_values[best_k]
                else:
                    # Compute the best gamma value for the current alpha checking for all the previous test samples and for all possible gamma
                    minval = np.min(np.abs((1 - alpha) - np.mean(valid_intervals[alphaidx, :(t_s), range(K)], axis=1)))
                    best_k = np.max(np.where(
                        np.abs((1 - alpha) - np.mean(valid_intervals[alphaidx, :(t_s), range(K)], axis=1)) == minval))
                    # Update the best gamma value for the current alpha
                    gamma_best_k[alphaidx] = gamma_values[best_k]

            # Apply naive algorithm based on asymptotic conformal inference to compute PIs
            test_PIs.append(build_naci_pis(preds_cali=preds_cali, y_actual=y_actual, y_cali=y_cali,
                                          preds_test=preds_test, gamma_best_k=gamma_best_k,
                                          tsample=t_s, gamma_values=gamma_values, alphagamma=alphagamma,
                                          interval_lengths=interval_lengths, valid_intervals=valid_intervals,
                                          settings=settings))

    # Concatenate the PIs for each test sample
    test_PIs = np.concatenate(test_PIs, axis=0)
    # Build updated dataframe
    aggr_df = recalib_preds.filter([settings['task_name']], axis=1)
    aggr_df = aggr_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]

    for j in range(len(settings['target_quantiles'])):
        aggr_df[settings['target_quantiles'][j]] = test_PIs[:, j]

    return aggr_df



def build_agaci_pis(preds_cali: np.array, y_cali: np.array, y_actual: np.array, preds_test: np.array, tsample: int,
                    gamma_values: np.array, alphagamma: np.array, upperbounds: np.array, lowerbounds: np.array,
                    yactualuptonow: np.array, settings: Dict, method: str = 'higher'):
    '''
    Compute PIs at the different alpha levels using Online Aggregation for ACI
    '''
    # Create a temporary directory to save CSV file
    temp_dir = 'C:\\Users\\Luigi\\Documenti\\Rstudio'   # Change the directory to your pc
    os.makedirs(temp_dir, exist_ok=True)
    # Squeeze the recalibration predictions
    preds_cali = np.squeeze(preds_cali, axis=-1)
    if preds_test.shape[0] > 1:
        sys.exit('ERROR: exec_cup supports single test samples')
    # Compute conformity score (absolute residual)
    conf_score = np.abs(preds_cali - y_cali)
    # Evaluate the number of available scores
    n = conf_score.shape[0]
    preds_test_q = [preds_test]
    # Initialize the matrix of aggregation weights
    # wu=np.zeros((tsample+1,len(gamma_values),len(settings['target_alpha'])))
    wu = np.zeros((len(gamma_values), len(settings['target_alpha'])))
    # wl = np.zeros((tsample+1,len(gamma_values), len(settings['target_alpha'])))
    wl = np.zeros((len(gamma_values), len(settings['target_alpha'])))
    # Store the target values from the beginning of the test sample
    yactualuptonow[:, tsample] = y_actual.reshape(-1, 1)[:, 0]
    # Stack the quantiles to the point prediction for each alpha and each gamma
    for gammaidx, gamma in enumerate(gamma_values):
        for alphaidx, alpha in enumerate(alphagamma[:, gammaidx]):
            # Compute the quantile for the current alpha
            q = np.ceil((n + 1) * (1 - alpha)) / n
            Q_1_alpha = np.expand_dims(np.quantile(a=conf_score, q=q, axis=0, method=method), axis=(0, -1))
            # Check if the proportion of hourly target values outside the PI is >= the original alpha

            if np.sum(np.logical_or(y_actual.reshape(-1, 1) >= np.squeeze((preds_test + Q_1_alpha), axis=0),
                                    y_actual.reshape(-1, 1) <= np.squeeze((preds_test - Q_1_alpha), axis=0))) / \
                    settings['pred_horiz'] >= settings['original_alpha'][alphaidx]:
                # Update the alpha value
                alphagamma[alphaidx, gammaidx] += gamma * (settings['original_alpha'][alphaidx] - 1)
            else:
                # Update the alpha value
                alphagamma[alphaidx, gammaidx] += gamma * settings['original_alpha'][alphaidx]
            # Store all the previous upper and lower bounds adapting properly the dimension
            upperbounds[alphaidx, :, tsample, gammaidx] = np.transpose(np.squeeze((preds_test + Q_1_alpha), axis=0))
            lowerbounds[alphaidx, :, tsample, gammaidx] = np.transpose(np.squeeze((preds_test - Q_1_alpha), axis=0))

    for alphaidx, alpha in enumerate(settings['target_alpha']):
        #Adapt the dimension to that required by R
        r_Y = (yactualuptonow[:, :(tsample + 1)]).reshape(tsample + 1, settings['pred_horiz'])
        r_experts_low = lowerbounds[alphaidx, :, :(tsample + 1), :].reshape(tsample + 1, settings['pred_horiz'],
                                                                            len(gamma_values))
        r_experts_high = upperbounds[alphaidx, :, :(tsample + 1), :].reshape(tsample + 1, settings['pred_horiz'],
                                                                             len(gamma_values))
        # Save the data in CSV files starting from 3 numpy arrays
        r_Y_path = os.path.join(temp_dir, 'r_Y.csv')
        r_experts_low_path = os.path.join(temp_dir, 'r_experts_low.csv')
        r_experts_high_path = os.path.join(temp_dir, 'r_experts_high.csv')

        pd.DataFrame(r_Y).to_csv(r_Y_path, header=False, index=False)
        pd.DataFrame(r_experts_low.reshape(-1, r_experts_low.shape[-1])).to_csv(r_experts_low_path, header=False,
                                                                                index=False)
        pd.DataFrame(r_experts_high.reshape(-1, r_experts_high.shape[-1])).to_csv(r_experts_high_path, header=False,
                                                                                  index=False)

        # Execute the R function aggregate_boas.R via the Pycharm Plugin R LANGUAGE FOR INTELLIJ                                                                       index=False)

        cmd = [
            'Rscript', 'aggregate_boas.R',
            r_Y_path, r_experts_low_path, r_experts_high_path, str(settings['original_alpha'][alphaidx]),
            str(r_experts_low.shape[0]), str(r_experts_low.shape[1]), str(r_experts_low.shape[2])
        ]

        # Convert the txt file in np.array
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return
        rows = result.stdout.strip().split('\n')[1:]

        #extract the last row, that of weights at time t_sample_test
        weights = []
        for row in rows:
            values = row.strip().split()
            if len(values) > 1:
                try:
                    weights.append([float(val) for val in values[1:]])
                except ValueError:
                    pass

        weights = np.array(weights)
        weights_shape = weights.shape[0]
        # Separate between the lower weights and the upper weights
        wl[:, alphaidx] = weights[(weights_shape // 2) - 1, :]
        wu[:, alphaidx] = weights[weights.shape[0] - 1, :]

        # Append lower/upper PIs for the current alpha according to the weights
        preds_test_q.append(np.expand_dims(np.transpose(np.dot(np.transpose((wu[:, alphaidx]).reshape(-1, 1)),
                                                               (np.transpose(upperbounds[alphaidx, :, tsample, :])))/np.sum(wu[:, alphaidx])),
                                           axis=0))
        preds_test_q.append(np.expand_dims(np.transpose(np.dot(np.transpose((wl[:, alphaidx]).reshape(-1, 1)),
                                                               (np.transpose(lowerbounds[alphaidx, :, tsample, :])))/np.sum(wl[:, alphaidx])),
                                           axis=0))
    # Concatenate the PIs for each alpha
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    # Fix quantile crossing by sorting and return prediction flattened in temporal dimension (sample over pred horizon)
    return np.sort(preds_test_q.reshape(-1, preds_test_q.shape[-1]), axis=-1)


def compute_agaci(recalib_preds, settings):
    '''
    Reshape recalibration predictions and execute Online Aggregation on ACI for each test sample
    '''
    # Reshape recalibration predictions
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    ens_p = recalib_preds.loc[:, 0.5].to_numpy()
    ens_p_d = ens_p.reshape(-1, settings['pred_horiz'], 1)
    # Extract the target data
    target_d = recalib_preds.filter([settings['task_name']], axis=1).to_numpy().reshape(-1, settings['pred_horiz'])
    # Evaluate the number of remaining test samples
    num_test_samples = ens_p_d.shape[0] - settings['num_cali_samples']
    # Create an empty list to store the target values for each test sample
    yactualuptonow = np.zeros((settings['pred_horiz'], num_test_samples))
    test_PIs = []
    # Save the original alpha values
    settings['original_alpha'] = settings['target_alpha']
    # Save the gamma values
    gamma_values = settings['lr']
    # Save the number of gamma values
    K = len(gamma_values)
    # Initialize the matrix of alpha values for different gamma
    alphagamma = np.zeros((len(settings['target_alpha']), K))
    # Create the tensor of upper and lower bounds for each alpha and gamma saving at each test sample even the results for the previous ones
    lowerbounds = np.zeros((len(settings['target_alpha']), settings['pred_horiz'], num_test_samples, K))
    upperbounds = np.zeros((len(settings['target_alpha']), settings['pred_horiz'], num_test_samples, K))
    # Initialize with the original values of alpha
    for k in range(K):
        alphagamma[:, k] = settings['target_alpha']

    for t_s in range(num_test_samples):
        # Extract settings['num_cali_samples'] recalibration predictions
        preds_cali = ens_p_d[t_s:settings['num_cali_samples'] + t_s]
        # Extract the test prediction for the current test sample
        preds_test = ens_p_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s + 1]
        # Extract settings['num_cali_samples'] target values
        y_cali = target_d[t_s:settings['num_cali_samples'] + t_s]
        # Extract the target value for the current test sample
        y_actual = target_d[settings['num_cali_samples'] + t_s]
        # Apply Online Aggregation on ACI to compute PIs
        test_PIs.append(
            build_agaci_pis(preds_cali=preds_cali, y_cali=y_cali, y_actual=y_actual, preds_test=preds_test, tsample=t_s,
                            gamma_values=gamma_values, alphagamma=alphagamma, upperbounds=upperbounds,
                            lowerbounds=lowerbounds, yactualuptonow=yactualuptonow, settings=settings))
    # Concatenate the PIs for each test sample
    test_PIs = np.concatenate(test_PIs, axis=0)
    # Build updated dataframe
    aggr_df = recalib_preds.filter([settings['task_name']], axis=1)
    aggr_df = aggr_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]

    for j in range(len(settings['target_quantiles'])):
        aggr_df[settings['target_quantiles'][j]] = test_PIs[:, j]

    return aggr_df
