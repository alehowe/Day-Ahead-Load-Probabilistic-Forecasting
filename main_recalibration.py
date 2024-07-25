"""
Main script to run the recalibration experiments
"""

# Author: Alessandro Howe

import os
import pandas as pd
import numpy as np
os.environ["TF_USE_LEGACY_KERAS"]="1"
from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs
from tools.prediction_quantiles_tools import plot_quantiles
import tensorflow as tf

from tools.conformal_prediction import compute_cp
from tools.conformal_prediction import compute_cqr
from tools.conformal_prediction import compute_aci
from tools.conformal_prediction import compute_naci
from tools.conformal_prediction import compute_agaci

from tools.data_utils import data_plotting
from tools.data_utils import data_testing
from tools.data_utils import remove_highly_correlated_features
from tools.data_utils import HolidayAnalyzer

from tools.scores import Losses
from tools.scores import compute_pinball_scores
from tools.scores import compute_winkler_scores
from tools.scores import compute_delta_coverage
from tools.scores import compute_scores_totquant
from tools.scores import compute_scores_tottime
from tools.scores import compute_pinball_score_perday
from tools.scores import compute_winkler_score_perday
from tools.scores import compute_avg_scores_
from tools.scores import plot_scores

import logging
import warnings
logging.captureWarnings(True)
warnings.filterwarnings('ignore', category=DeprecationWarning)

#--------------------------------------------------------------------------------------------------------------------
# Set PEPF task to execute

# set in Json TrainingAndProtoTest for testing
PF_task_name = 'NetLoad'  # or NetLoad

# Set Model setup to execute
exper_setup = 'point-CNNMIXED'

#---------------------------------------------------------------------------------------------------------------------
# Set run configs
run_id = 'recalib_opt_grid_1_1'

# Load hyperparams from file (select: load_tuned or optuna_tuner)
hyper_mode = 'load_tuned'   # change optuna_tuner or load_tuned

# Plot train history flag
plot_train_history=False
plot_weights=False

# Redefine holidays
redifine_holidays = True

# Transform the target variable if requested and standardize it if requested
transform_targ = "log"
standardize_asinh = True

# Choose whether to work out conformal predictions
exec_CP = True

# Number of calibration samples for conformal prediction ( < Number of predictions)
num_cali_samples = 365

# Flag that indicates which type of conformal prediction to perform
# "cp":     conformal prediction
# "cqr":    conformalized quantile regression for each quantile (assuming already having quantiles)
# "aci":    adaptive conformal inference
# "naci":   naive adaptive conformal inference
# "agaci":  Online Aggregation Adaptive Conformal Inference

type_conformal = "aci"

# Choose whether to compute the point scores
point_scores = True

# Choose whether to compute other scores
probabilistic_scores = True

#---------------------------------------------------------------------------------------------------------------------
# Load experiments configuration from json file
configs=load_data_model_configs(task_name=PF_task_name, exper_setup=exper_setup, run_id=run_id)
# Load dataset
dir_path = os.getcwd()

ds = pd.read_csv(os.path.join(dir_path, 'data', 'datasets', configs['data_config'].dataset_name))
ds.set_index(ds.columns[0], inplace=True)

#---------------------------------------------------------------------------------------------------------------------

# Drop leap years of the dataset to preserve seasonality
ds['Date'] = pd.to_datetime(ds['Date'])
ds = ds[(ds['Date'].dt.month!= 2) | (ds['Date'].dt.day!= 29)]
ds['Date'] = ds['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
ds.reset_index(drop=True, inplace=True)

# Plot some dataset features
plot_features=data_plotting(ds, configs)
plot_features.plot_overall()
plot_features.plot_month_hour_seasonality(1)
plot_features.plot_weekday_weekend_hour_seasonality(7, 2015)
plot_features.plot_monthly_netload_data(months=['01', '04', '07', '10'])
plot_features.plot_monthly_heatmap(['01', '04', '07', '10'])
plot_features.plot_netload_days(['01', '04', '07', '10'],['01', '05', '10', '15', '20', '25', '30'])
plot_features.plot_autocorrelation()


# Aggregate the sd information in a unique variable representing overall weather volatility
futu_sd_cols = [col for col in ds.columns if col.endswith('Sd')]
ds['FUTU__OverallSd'] = ds[futu_sd_cols].mean(axis=1)
pattern = '|'.join(futu_sd_cols)
ds.drop(ds.filter(regex=pattern).columns.tolist(), axis=1, inplace=True)

# Plot futu autocorrelation
plot_features=data_plotting(ds, configs)
plot_features.plot_futu_correlation()

# Look at the correlation of each feature variable and the target
correlation_matrix = ds.corr()
target_correlation = correlation_matrix["TARG__NetLoad"]
#print(target_correlation)

# Remove very highly correlated variables from the dataset
threshold = 0.9
columns_to_keep = remove_highly_correlated_features(ds[(ds['Date'] < pd.to_datetime(configs['data_config'].idx_start_oos_preds))].copy())

# Keep amongst the 2 wind speed mean variables the one more correlated with the target
if 'FUTU__WindSpeedAt100mRegionalMean' in columns_to_keep and 'FUTU__WindSpeedAt10mRegionalMean' not in columns_to_keep:
    columns_to_keep_list = list(columns_to_keep)
    columns_to_keep_list.remove('FUTU__WindSpeedAt100mRegionalMean')
    columns_to_keep_list.append('FUTU__WindSpeedAt10mRegionalMean')
    columns_to_keep = pd.Index(columns_to_keep_list)

ds = ds[columns_to_keep]

# Aggregate other variables (cancel this if not needed)
#ds.drop('FUTU__MediumCloudCoverageRegionalMean', axis=1, inplace=True)
#ds.drop('FUTU__HighCloudCoverageRegionalMean', axis=1, inplace=True)

correlation_matrix = ds.corr()
target_correlation = correlation_matrix["TARG__NetLoad"]


# Normality test on target variable
testing=data_testing(ds, configs)
testing.target_normality()

#---------------------------------------------------------------------------------------------------------------------
# Holiday redefinition
if redifine_holidays:
    holiday_an = HolidayAnalyzer(ds, 2016)
    ds = holiday_an.analyze()

#---------------------------------------------------------------------------------------------------------------------
# Transform the target variable if requested and standardize it if requested

# log transformation
if transform_targ == "log":

    ds["TARG__NetLoad"] = np.log((ds["TARG__NetLoad"]))

    # Test normality of the new target variable
    testing=data_testing(ds, configs)
    testing.target_normality()

# asinh transformation
elif transform_targ == "asinh":

    if standardize_asinh:
        mean_price = np.mean(ds["TARG__NetLoad"])
        std_price = np.std(ds["TARG__NetLoad"])
        ds["TARG__NetLoad"] = np.arcsinh((ds["TARG__NetLoad"]-mean_price)/std_price )
    else:
        ds["TARG__NetLoad"] = np.arcsinh((ds["TARG__NetLoad"]))

    # Test normality of the new target variable
    testing = data_testing(ds, configs)
    testing.target_normality()

#---------------------------------------------------------------------------------------------------------------------
# Instantiate recalibratione engine
PrTSF_eng = PrTsfRecalibEngine(dataset = ds,
                               data_configs = configs['data_config'],
                               model_configs=configs['model_config'],
                               )

# Get model hyperparameters (previously saved or by tuning)
model_hyperparams = PrTSF_eng.get_model_hyperparams(method=hyper_mode, optuna_m=configs['model_config']['optuna_m'])

# Exec recalib loop over the test_set samples, using the tuned hyperparams
test_predictions = PrTSF_eng.run_recalibration(model_hyperparams=model_hyperparams,
                                               plot_history=plot_train_history,
                                               plot_weights=plot_weights)


# Retransform the dataset
# log transformation
if transform_targ == "log":

    # Perform the reverse transformation
    test_predictions = np.exp(test_predictions)

# asinh transformation
elif transform_targ == "asinh":

    if standardize_asinh:
        # Multiply all elements by std_price and then add mean_price to each element
        test_predictions=mean_price+np.sinh(test_predictions)*std_price
    else:
        # Just perform the reverse transformation
        test_predictions=np.sinh(test_predictions)


#--------------------------------------------------------------------------------------------------------------------
# Conformal prediction settings

#cp_settings={'target_alpha':[0.10]}
#num_cali_samples = 31
#cp_settings = {'target_alpha': [0.09, 0.10]}
#num_cali_samples = 12


if exec_CP:
    # set the size of the calibration set sufficiently large to cover the target alpha (tails)
    cp_settings = {'target_alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]}


    # build the settings to build PF from point using CP
    cp_settings['pred_horiz']=configs['data_config'].pred_horiz
    cp_settings['task_name']=configs['data_config'].task_name
    cp_settings['num_cali_samples']=num_cali_samples

    if type_conformal=="cp":
       # exec conformal prediction
        test_predictions = compute_cp(test_predictions,cp_settings)

    elif type_conformal == "aci":
        # exec adaptive conformal prediction
        cp_settings['lr'] = 0.0001
        test_predictions = compute_aci(test_predictions,cp_settings)

    elif type_conformal == "naci":
        # Store hyper-parameters
        cp_settings['warm_up_period'] = 80
        cp_settings['lr'] = [0.0005, 0.001]

        # exec conformal prediction
        test_predictions = compute_naci(test_predictions, cp_settings)

    elif type_conformal == "agaci":
        # Store hyper-parameters
        cp_settings['lr'] = [0.0005, 0.001]

        # exec conformal prediction
        test_predictions = compute_agaci(test_predictions, cp_settings)

    elif type_conformal == "cqr":
        # exec conformalized quantile regression for each quantile
        test_predictions = compute_cqr(test_predictions, cp_settings)

    else:
        print('Conformal prediction NOT implemented')


#--------------------------------------------------------------------------------------------------------------------
# Compute delta coverage scores

if exec_CP:
    quantiles_levels = PrTSF_eng.__build_target_quantiles__(cp_settings['target_alpha'])
else:
    quantiles_levels = PrTSF_eng.model_configs['target_quantiles']

pred_steps = configs['model_config']['pred_horiz']

compute_delta_coverage(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1, pred_steps),
                           pred_quantiles=test_predictions.loc[:,
                                          test_predictions.columns != PF_task_name].to_numpy().reshape(-1, pred_steps,
                                                                                                       len(quantiles_levels)),
                           quantiles_levels=quantiles_levels,
                           alpha_min=0.90,
                           alpha_max=0.99,
                           configs = configs)

#--------------------------------------------------------------------------------------------------------------------
# Compute other probabilistic scores and plot them

if (probabilistic_scores):

    pinball_scores = compute_pinball_scores(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                            pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                            to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                            quantiles_levels=quantiles_levels)

    # Ensemble of the scores
    avg_pinball_score_time=compute_scores_tottime(pinball_scores)
    avg_pinball_score_quant=compute_scores_totquant(pinball_scores)
    avg_pinball_score_perday=compute_pinball_score_perday(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                            pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                            to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                            quantiles_levels=quantiles_levels)
    avg_pinball_score=compute_avg_scores_(pinball_scores)

    # Compute winkler scores
    winkler_scores= compute_winkler_scores(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                            pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                            to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                            quantiles_levels=quantiles_levels)

    # Compute ensemble of winkler scores
    avg_winkler_score_time=compute_scores_tottime(winkler_scores)
    avg_winkler_score_quant=compute_scores_totquant(winkler_scores)
    avg_winkler_score_perday=compute_winkler_score_perday(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                            pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                            to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                            quantiles_levels=quantiles_levels)

    avg_winkler_score=compute_avg_scores_(winkler_scores)

    # Print the scores
    #print("The pinball scores are: ", pinball_scores)
    #print("The average pinball scores for each time step are: ", avg_pinball_score_quant)
    #print("The average pinball scores for each quantile are: ", avg_pinball_score_time)
    #print("The average pinball scores per day is: ", avg_pinball_score_perday)
    print("The average pinball score is:", avg_pinball_score)

    #print("The winkler scores are: ", winkler_scores)
    #print("The average winkler scores for each time step are: ", avg_winkler_score_quant)
    #print("The average winkler scores for each quantile are: ", avg_winkler_score_time)
    #print("The average winkler scores per day is: ", avg_winkler_score_perday)
    print("The average winkler score is:", avg_winkler_score)

    # Plot the scores
    plot_scores(avg_pinball_score_quant, "Pinball Score", configs, "Score_per_hour")
    plot_scores(avg_pinball_score_perday, "Pinball Score", configs, "Score_per_day")

    plot_scores(avg_winkler_score_quant, "Winkler Score", configs, "Score_per_hour")
    plot_scores(avg_winkler_score_perday, "Winkler Score", configs, "Score_per_day")

#--------------------------------------------------------------------------------------------------------------------
# Compute point scores

if point_scores:
    # Retrieve the values
    true_y = tf.cast(test_predictions['NetLoad'], tf.float64)
    pred_y = tf.cast(test_predictions[0.5], tf.float64)

    # Compute the losses
    losses = Losses(true_y, pred_y)

    # Print them
    print("\nMSE Average:", losses.mse_loss_avg(), "MAE Average:", losses.mae_loss_avg(), "\nRMSE Average:",
          losses.rmse_loss_avg(), "\nsMAPE Average:", losses.sMAPE_loss_avg(),
          "\nMSE Hourly:", losses.mse_loss_hourly(), "\nMAE Hourly:", losses.mae_loss_hourly(), "\nRMSE Hourly:",
          losses.rmse_loss_hourly(), "\nsMAPE Hourly:", losses.sMAPE_loss_hourly())

    plot_scores(losses.mse_loss_daily(), "MSE",  configs, "Score_per_day")
    plot_scores(losses.rmse_loss_daily(), "RMSE", configs, "Score_per_day")
    plot_scores(losses.mae_loss_daily(), "MAE", configs, "Score_per_day")
    plot_scores(losses.sMAPE_loss_daily(), "MAPE", configs, "Score_per_day")


#--------------------------------------------------------------------------------------------------------------------
# Plot test predictions
plot_quantiles(test_predictions, target=PF_task_name)

#--------------------------------------------------------------------------------------------------------------------
print('Done!')