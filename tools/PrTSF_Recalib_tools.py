"""
Main tools managing the recalibration process
"""

import os
import sys
from datetime import date
from datetime import datetime
from typing import Dict, List, Union
import re
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
import tensorflow as tf

from tools.data_utils import columns_keys, features_keys
from tools.models.models_tools import regression_model, Ensemble, get_model_class_from_conf

from tools.models.GLM import GLM
from tools.VariableSelection import VariableSelection
from tools.data_utils import fourier_transform


class RecalibBlock:
    """
    Class used to structure train/vali samples related to each recalibration block
    """

    def __init__(self, x_train, y_train, x_vali, y_vali):
        self.x_train = x_train
        self.y_train = y_train
        self.x_vali = x_vali
        self.y_vali = y_vali


class RecalibSamples:
    """
    Class used to structure the recalibration samples, including the test sample and the related list of recalib blocks
    """

    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        self.recalibBlocks = []

    def add_recal_block(self, x_train, y_train, x_vali, y_vali):
        self.recalibBlocks.append(RecalibBlock(x_train=x_train, y_train=y_train,
                                               x_vali=x_vali, y_vali=y_vali))


class WindowGenerator:
    """
    Creates the shifting windows, following the approach reported in the TF docs
    """

    def __init__(self,
                 input_width: int,
                 label_width: int,
                 shift: int,
                 data_columns: List,
                 target_columns: List = None,
                 true_columns: List = None):

        # Work out the label column indices.
        self.label_columns = target_columns
        if target_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(target_columns)}
        self.true_columns = true_columns

        if true_columns is not None:
            self.true_columns_indices = {name: i for i, name in
                                         enumerate(true_columns)}

        self.column_indices = {name: i for i, name in
                               enumerate(data_columns)}

        # Store the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        # create slice object
        # To include the future conditioning features, the input slide include the prediction steps
        # The target column MUST be removed during input conditioning construction
        self.input_slice = slice(0, input_width + label_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        if target_columns is not None:
            self.true_start = self.label_start
            self.true_slice = self.labels_slice
            self.true_indices = self.label_indices

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}',
            f'True indices: {self.true_indices}',
            f'True column name(s): {self.true_columns}'])

    def split_window(self, features):

        inputs = features[:, self.input_slice, :]
        targets = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            targets = np.stack(
                [targets[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        return inputs, np.squeeze(targets, axis=-1)


class PrTsfDataloaderConfigs:
    """
    Class used to handle the dataloader configuration
    """

    def __init__(self,
                 task_name: str,
                 exper_setup: str,
                 dataset_name: str,
                 idx_start_train: Union[int, date],
                 idx_start_oos_preds: Union[int, date],
                 idx_end_oos_preds: Union[int, date],
                 num_vali_samples: int = 180,
                 steps_lag_win: int = 7,
                 pred_horiz: int = 24,
                 preprocess: str = 'StandardScaler',
                 keep_past_train_samples: bool = True,
                 shuffle_mode: str = 'none',
                 per_max: int = 10000,
                 per_min: int = 1,
                 freq: int = 3,
                 analysis: str = 'None',
                 pca: bool = False,
                 holiday : str = 'None',
                 periodic: bool = False,
                 holiday_distance: bool = False
                 ):
        self.task_name = task_name
        self.exper_setup = exper_setup
        self.dataset_name = dataset_name
        self.idx_start_train = idx_start_train
        self.idx_start_oos_preds = idx_start_oos_preds
        self.idx_end_oos_preds = idx_end_oos_preds
        self.num_vali_samples = num_vali_samples
        self.steps_lag_win = steps_lag_win
        self.pred_horiz = pred_horiz
        self.preprocess = preprocess
        self.keep_past_train_samples = keep_past_train_samples
        self.shuffle_mode = shuffle_mode
        self.per_max = per_max
        self.per_min = per_min
        self.freq = freq
        self.analysis = analysis
        self.pca = pca
        self.holiday = holiday
        self.periodic = periodic
        self.holiday_distance = holiday_distance



def load_data_model_configs(task_name: str, exper_setup: str, run_id: str):
    """
    Load experiment configurations from json and build the handler object
    """
    path = os.path.join(os.getcwd(), 'experiments', 'tasks', task_name, exper_setup, run_id, 'exper_configs.json')

    # Load experiment settings from json
    with open(path) as f:
        expe_confs = json.load(f)

    expe_confs['data_config']['idx_start_train'] = date(year=expe_confs['data_config']['idx_start_train']['y'],
                                                        month=expe_confs['data_config']['idx_start_train']['m'],
                                                        day=expe_confs['data_config']['idx_start_train']['d'])
    expe_confs['data_config']['idx_start_oos_preds'] = date(year=expe_confs['data_config']['idx_start_oos_preds']['y'],
                                                            month=expe_confs['data_config']['idx_start_oos_preds']['m'],
                                                            day=expe_confs['data_config']['idx_start_oos_preds']['d'])
    expe_confs['data_config']['idx_end_oos_preds'] = date(year=expe_confs['data_config']['idx_end_oos_preds']['y'],
                                                          month=expe_confs['data_config']['idx_end_oos_preds']['m'],
                                                          day=expe_confs['data_config']['idx_end_oos_preds']['d'])
    # Store exper run id
    expe_confs['model_config']['run_id'] = run_id

    # Append running experiments configs
    data_configs = PrTsfDataloaderConfigs(
        task_name=task_name,
        exper_setup=exper_setup,
        dataset_name=expe_confs['data_config']['dataset_name'],
        idx_start_train=expe_confs['data_config']['idx_start_train'],
        idx_start_oos_preds=expe_confs['data_config']['idx_start_oos_preds'],
        idx_end_oos_preds=expe_confs['data_config']['idx_end_oos_preds'],
        num_vali_samples=expe_confs['data_config']['num_vali_samples'],
        steps_lag_win=expe_confs['data_config']['steps_lag_win'],
        pred_horiz=expe_confs['data_config']['pred_horiz'],
        preprocess=expe_confs['data_config']['preprocess'],
        keep_past_train_samples=expe_confs['data_config']['keep_past_train_samples'],
        shuffle_mode=expe_confs['data_config']['shuffle_mode'],
        per_max=expe_confs['data_config']['per_max'],
        per_min=expe_confs['data_config']['per_min'],
        freq=expe_confs['data_config']['freq'],
        analysis=expe_confs['data_config']['analysis'],
        pca=expe_confs['data_config']['pca'],
        holiday=expe_confs['data_config']['holiday'],
        periodic=expe_confs['data_config']['periodic'],
        holiday_distance = expe_confs['data_config']['holiday_distance']
    )
    return {'data_config': data_configs,
            'model_config': expe_confs['model_config']}


class PrTsfRecalibEngine:
    """
    Main class executing the recalibration process
    """

    def __init__(self, dataset,
                 data_configs: PrTsfDataloaderConfigs,
                 model_configs: Dict):
        self.data_configs = data_configs
        self.data_transf = self.data_configs.pca
        self.holiday = self.data_configs.holiday
        self.dataset = dataset.copy()
        self.periodic = self.data_configs.periodic
        # store the samples involved in the configured experimental period (between start_train and oos_end) and reindex
        self.__store_reindexed_dataset__(data_configs=data_configs)
        # build test samples idxs used by the recalibration iterator
        self.test_set_idxs = self.__build_test_samples_idxs__()
        # instantiate preprocessing_objs
        self.preproc = self.__instantiate_preproc__()

        # Variable selection of the dataset
        self.original_dataset = self.dataset.copy()

        # If required, perform PCA / variable selection (see VariableSelection class)
        if self.data_transf:
            dataset = self.original_dataset.copy()
            SelectionProcess = VariableSelection(dataset, self.data_configs.idx_start_oos_preds, self.data_configs,
                                                 self.preproc)
            self.dataset = SelectionProcess.selection()
            self.dataset['Date'] = self.original_dataset['Date'].copy()
            self.dataset['Hour'] = self.original_dataset['Hour'].copy()


        # store model configs and add internal confs automatically
        self.model_configs = model_configs
        self.model_class = get_model_class_from_conf(self.model_configs['model_class'])
        # Copy pred_horizon from data confs
        self.model_configs['pred_horiz'] = self.data_configs.pred_horiz
        # Build target quantiles from alpha, including the median
        self.model_configs['target_quantiles'] = self.__build_target_quantiles__(self.model_configs['target_alpha'])
        # Build mapping between quantile idx and alpha/median
        self.model_configs['q_alpha_map'] = self.__build_alpha_quantiles_map__(
            target_quantiles=self.model_configs['target_quantiles'],
            target_alpha=self.model_configs['target_alpha'])
        # Build wavelength searchspace for seasonality
        self.model_configs['period min'] = self.data_configs.per_min
        self.model_configs['period max'] = self.data_configs.per_max
        # Define number of frequencies to use in fourier transform
        self.model_configs['freq'] = self.data_configs.freq
        self.model_configs['analysis'] = self.data_configs.analysis
        self.model_configs['holiday_distance'] = self.data_configs.holiday_distance

        # Define the class flags
        self.fourier_analysis = True if self.model_configs['analysis'] == 'Fourier' else False
        self.GLM_regressor = True if  self.model_configs['analysis'] == 'Glm' else False
        dataset_remove_seasonality = self.original_dataset.copy()

        # Dop the target value from the dataset used for the model if we are trying to remove the seasonality
        if (self.GLM_regressor or self.fourier_analysis):
            self.dataset.drop('TARG__NetLoad', axis=1, inplace=True)

        if (self.fourier_analysis):
            # Fit the model and predict over the entire testing dataset
            fourier_object = fourier_transform(self.model_configs['period min'], self.model_configs['period max'],
                                               self.model_configs['freq'])
            fourier_dataset = fourier_object.fit(
                training_dataset=dataset_remove_seasonality[:self.test_set_idxs[0]].copy())
            fourier_object.create_plots_fourier(fourier_dataset)
            fourier_residuals, self.fourier_baseline = fourier_object.predict(
                testing_dataset=dataset_remove_seasonality[
                                self.data_configs.idx_start_oos_preds:self.data_configs.idx_end_oos_preds + 1].copy())

            # Store the used values in the dataset used for the model
            self.dataset['TARG__NetLoad'] = fourier_residuals.values
            self.data_configs.preprocess = 'MinMaxScalerResidue'
            self.dataset['TRUE__NetLoad'] = self.original_dataset['TARG__NetLoad'].copy()

        if (self.GLM_regressor):
            init_sample = 0
            # Fit the model
            glm_regressor = GLM(start_index=init_sample)
            train_residuals = glm_regressor.fit_GLM(
                training_dataset=dataset_remove_seasonality[init_sample:self.data_configs.idx_start_oos_preds].copy())

            # Prediction on testing dataset
            self.y_hat, test_residuals = glm_regressor.predict_GLM(testing_dataset=dataset_remove_seasonality[
                                                                                   self.data_configs.idx_start_oos_preds: self.data_configs.idx_end_oos_preds + 1].copy())

            glm_regressor.residual_normality()
            # Conatenate the residuals
            self.residuals = pd.concat([train_residuals, test_residuals]).reset_index(drop=True)

            # Plots of the results of the GLM model
            glm_regressor.plot_residuals_GLM()
            glm_regressor.plot_prediction_GLM()

            # Store the used values in the dataset used for the model
            self.dataset.loc[init_sample: self.data_configs.idx_end_oos_preds,
            'TARG__Residuals'] = self.residuals.values
            self.dataset['TRUE__NetLoad'] = self.original_dataset['TARG__NetLoad'].copy()

        if self.periodic:
            # Transform the periodic variables with cos transformation
            if self.holiday == 'Dummy':
                raise ValueError("Holiday approach not compatible with date modeling")

            self.dataset['CONST__Hour'] = np.cos(2 * np.pi * self.dataset['Hour'] / 24)
            self.dataset['CONST__DoW'] = np.cos(2 * np.pi * self.dataset['CONST__DoW'] / 7)
            self.dataset['CONST__DoY'] = np.cos(2 * np.pi * self.dataset['CONST__DoY'] / 365)
            self.dataset['CONST__Month'] = np.cos(2 * np.pi * (self.dataset['Date'].dt.month - 1) / 12)

            if self.model_configs['holiday_distance'] == True:

                # Calculate forward distances to the next 1 in the 'CONST__Holiday' column
                n = len(self.dataset)
                distances = np.full(n, np.inf)
                next_holiday_distance = np.inf

                # Iterate backwards to propagate the distance to the nearest holiday forward
                for i in range(n - 1, -1, -1):
                    if self.dataset.at[i, 'CONST__Holiday'] == 1:
                        next_holiday_distance = 0
                    distances[i] = next_holiday_distance
                    next_holiday_distance += 1

                #  Holiday distance technique

                self.dataset['PAST__Distance_To_Holiday'] = distances
                self.dataset.drop(columns=['CONST__Holiday'], inplace=True)

        # Generate 24 dummy variables to encode the hour of the day and concatenate with the regressor dataframe
        else:
            df_hour_dummies = pd.get_dummies(self.dataset['Hour'].astype("category"), prefix='DUMMY__Hour',
                                             drop_first=False).astype(int)

            self.dataset = pd.concat([self.dataset, df_hour_dummies], axis=1)

            # Generate 2 dummy variables to encode the day of the week and concatenate with the regressor dataframe

            df_week_dummies = pd.get_dummies(self.dataset['CONST__DoW'].astype("category"), prefix='DUMMY__Day_of_Week',
                                             drop_first=False).astype(int)

            df_weekday = df_week_dummies.iloc[:, 1:-1].sum(axis=1)
            df_weekday.name = 'DUMMY__Weekday'

            df_weekend = df_week_dummies.iloc[:, 0] + df_week_dummies.iloc[:, -1]

            df_weekend.name = 'DUMMY__Weekend'

            if self.holiday == 'Dummy':
                # Apply the holiday condition
                df_weekday[self.dataset['CONST__Holiday'] == 1] = 0
                df_weekend[self.dataset['CONST__Holiday'] == 1] = 1
                self.dataset.drop('CONST__Holiday', inplace = True, axis = 1)

            self.dataset = pd.concat([self.dataset, df_weekend], axis=1)

            #Generate 12 dummy variables to encode the month of the year and concatenate with the regressor dataframe

            df_month_dummies = pd.get_dummies(self.dataset['Date'].dt.month.astype("category"),
                                              prefix='DUMMY__Month_of_Year', drop_first=False).astype(int)

            self.dataset = pd.concat([self.dataset, df_month_dummies], axis=1)


            self.dataset.drop(['IDX__step', 'IDX__sub_step', 'CONST__DoW', 'CONST__DoY'], inplace=True, axis=1)


    @staticmethod
    def __load_dataset_from_file__(dataset_name: str):
        """
        Load data from csv
        """
        dir_path = os.getcwd()
        ds = pd.read_csv(os.path.join(dir_path, 'data', 'datasets', dataset_name))
        ds.set_index(ds.columns[0], inplace=True)
        return ds

    def __get_global_idx_from_date__(self, date_id, mode='start'):
        """
        Get the global idx related to the input date.
        Mode: 'start': return the idx of first sub_step; 'end': return the idx of first sub_step
        """
        date_idxs = self.dataset[self.dataset[columns_keys['Date']] == date_id.strftime('%Y-%m-%d')].index.tolist()
        if mode == 'start':
            global_idx = date_idxs[0]
        elif mode == 'end':
            global_idx = date_idxs[-1]
        else:
            sys.exit('ERROR: selected mode do not exist')

        return global_idx

    def __store_reindexed_dataset__(self, data_configs: PrTsfDataloaderConfigs):
        """
        Get train/test periods from configs and store
        """
        if (type(data_configs.idx_start_train) is date and
                type(data_configs.idx_start_oos_preds) is date and
                type(data_configs.idx_end_oos_preds) is date):
            self.data_configs = data_configs
            # set idx from input date
            self.data_configs.idx_start_train = self.__get_global_idx_from_date__(self.data_configs.idx_start_train,
                                                                                  mode='start')
            self.data_configs.idx_start_oos_preds = self.__get_global_idx_from_date__(
                self.data_configs.idx_start_oos_preds, mode='start')
            self.data_configs.idx_end_oos_preds = self.__get_global_idx_from_date__(self.data_configs.idx_end_oos_preds,
                                                                                    mode='end')

        elif (type(data_configs.idx_start_train) is int and
              type(data_configs.idx_start_oos_preds) is int and
              type(data_configs.idx_end_oos_preds) is int):
            self.data_configs = data_configs
        else:
            sys.exit('ERROR: idx_start_train and idx_start_end can be either int or date vars!')

        # Extract dataset samples covering the experiment period

        self.dataset = self.dataset[self.data_configs.idx_start_train:self.data_configs.idx_end_oos_preds + 1]
        # Reindex dataset and store updated idxs in configs
        # self.dataset[columns_keys['idx_global']] = np.arange(len(self.dataset))
        # self.dataset[columns_keys['idx_step']] = np.arange(stop=len(self.dataset)) // self.data_configs.pred_horiz
        self.dataset.loc[:, [columns_keys['idx_global']]] = np.arange(len(self.dataset))
        self.dataset.loc[:, [columns_keys['idx_step']]] = np.arange(
            stop=len(self.dataset)) // self.data_configs.pred_horiz
        init_global_idx = self.dataset.index.tolist()[0]
        self.data_configs.idx_start_train = self.data_configs.idx_start_train - init_global_idx
        self.data_configs.idx_start_oos_preds = self.data_configs.idx_start_oos_preds - init_global_idx
        self.data_configs.idx_end_oos_preds = self.data_configs.idx_end_oos_preds - init_global_idx
        self.dataset.set_index(self.dataset[columns_keys['idx_global']], inplace=True)
        self.dataset = self.dataset.drop(columns=[columns_keys['idx_global']])

    def __build_test_samples_idxs__(self):
        return np.arange(start=self.data_configs.idx_start_oos_preds,
                         stop=self.data_configs.idx_end_oos_preds,
                         step=self.data_configs.pred_horiz)

    def __instantiate_preproc__(self):
        if self.data_configs.preprocess == 'StandardScaler':
            preproc = {
                'feat': StandardScaler(),
                'target': StandardScaler()
            }

        elif self.data_configs.preprocess == 'MinMaxScaler':
            preproc = {
                'feat': MinMaxScaler(feature_range=(0, 1)),
                'target': MinMaxScaler(feature_range=(0, 1))
            }
        elif self.data_configs.preprocess == 'MinMaxScalerResidue':
            preproc = {
                'feat': MinMaxScaler(feature_range=(0, 1)),
                'target': MinMaxScaler(feature_range=(-1, 1))
            }

        elif self.data_configs.preprocess == 'RobustScaler':
            preproc = {
                'feat': RobustScaler(),
                'target': RobustScaler()
            }

        elif self.data_configs.preprocess == 'QuantileTransformer':
            preproc = {
                'feat': QuantileTransformer(output_distribution='normal'),
                'target': QuantileTransformer(output_distribution='normal')
            }


        else:
            sys.exit('ERROR: selected preprocessing not implemented')

        return preproc

    def __build_recalib_dataset_batches__(self, df: pd.DataFrame, fit_preproc: bool) -> object:

        # extract features and target columns from the whole dataframe
        df_feat = df.filter(
            regex=features_keys['past'] + '|' + features_keys['futu'] + '|' + features_keys['const'] + '|' +
                  features_keys['pc'] + '|' + features_keys['idx'] + '|' + features_keys['dummy'])
        df_target = df.filter(regex=features_keys['target'])
        df_true = df.filter(regex=features_keys['true'])

        # Fit preprocessing objects using the series steps before the pred_horiz (i.e., the recalibration test sample)
        if fit_preproc:
            self.preproc['feat'].fit(df_feat[:-self.data_configs.pred_horiz])
            self.preproc['target'].fit(df_target[:-self.data_configs.pred_horiz])

        # Transform the series by preprocessing objects
        np_feat_scaled = self.preproc['feat'].transform(df_feat)
        np_target_scaled = self.preproc['target'].transform(df_target)

        # Build scaled df
        df_feat_scaled = pd.DataFrame(data=np_feat_scaled,
                                      index=df.index,
                                      columns=df_feat.columns)
        df_target_scaled = pd.DataFrame(data=np_target_scaled,
                                        index=df.index,
                                        columns=df_target.columns)
        if (self.GLM_regressor or self.fourier_analysis):
            df_true = pd.DataFrame(data=df_true,
                                   index=df.index,
                                   columns=df_true.columns)

            df_scaled = pd.concat([df_feat_scaled, df_target_scaled, df_true], axis=1)

        else:

            df_scaled = pd.concat([df_feat_scaled, df_target_scaled], axis=1)

        # store x columns names
        self.x_columns_names = df_scaled.columns.tolist()
        self.model_configs['x_columns_names'] = self.x_columns_names

        # Create object used to generate samples following standard moving window
        target_col_name = [x for x in df_scaled.columns.tolist() if re.search(features_keys['target'], x)]
        true_col_name = [x for x in df_scaled.columns.tolist() if re.search(features_keys['true'], x)]

        self._win_gen = WindowGenerator(
            input_width=self.data_configs.steps_lag_win * self.data_configs.pred_horiz,
            label_width=self.data_configs.pred_horiz,
            shift=self.data_configs.pred_horiz,
            data_columns=df_scaled.columns,
            target_columns=target_col_name,
            true_columns=true_col_name)

        # Convert the series into samples
        series_np = np.array(df_scaled.values).astype(np.float32)
        series_samples = np.stack([series_np[i:i + self._win_gen.total_window_size] for i in
                                   range(0, series_np.shape[0] - self._win_gen.total_window_size + 1,
                                         self._win_gen.label_width)])

        # Extract the last sample for test (by step-wise recalibration
        recalib_test_sample = np.copy(series_samples[-1:])
        # Put the other samples in the trainvali bag
        recalib_trainvali_samples = np.copy(series_samples[:-1])

        # Shuffle trainvali samples if requested
        if self.data_configs.shuffle_mode == 'train_vali':
            np.random.shuffle(recalib_trainvali_samples)

        # Build input/output samples for train_vali and test
        trainvali_samples_x, trainvali_samples_y = self._win_gen.split_window(recalib_trainvali_samples)

        x_test, y_test = self._win_gen.split_window(recalib_test_sample)

        # Separate samples devoted to train and vali
        x_train = np.copy(trainvali_samples_x[:-self.data_configs.num_vali_samples])
        y_train = np.copy(trainvali_samples_y[:-self.data_configs.num_vali_samples])
        vali_samples_x = np.copy(trainvali_samples_x[-self.data_configs.num_vali_samples:])
        vali_samples_y = np.copy(trainvali_samples_y[-self.data_configs.num_vali_samples:])

        # shuffle vali samples if required
        if self.data_configs.shuffle_mode == 'vali':
            p = np.random.permutation(len(vali_samples_y))
            vali_samples_x = vali_samples_x[p]
            vali_samples_y = vali_samples_y[p]

        # Instantiate recalibration object
        rec_samples = RecalibSamples(x_test=x_test, y_test=y_test)
        rec_samples.add_recal_block(x_train=x_train,
                                    y_train=y_train,
                                    x_vali=vali_samples_x,
                                    y_vali=vali_samples_y)

        return rec_samples

    @staticmethod
    def __build_target_quantiles__(target_alpha: List):
        """
        Build target quantiles from the list of alpha, including the median
        """
        target_quantiles = [0.5]
        for alpha in target_alpha:
            target_quantiles.append(alpha / 2)
            target_quantiles.append(1 - alpha / 2)
        target_quantiles.sort()
        return target_quantiles

    @staticmethod
    def __build_alpha_quantiles_map__(target_alpha: List, target_quantiles: List):
        """
        Build the map between the alpha coverage levels and the related quantiles
        """
        alpha_q = {'med': target_quantiles.index(0.5)}
        for alpha in target_alpha:
            alpha_q[alpha] = {
                'l': target_quantiles.index(alpha / 2),
                'u': target_quantiles.index(1 - alpha / 2),
            }
        return alpha_q

    def __transform_test_results__(self, results_df: pd.DataFrame) -> object:
        """
        Create datetime object till the end of last test date to setup the date_range properly
        """
        date_format = '%Y-%m-%d %H:%M'

        end_date = self.original_dataset.iloc[self.data_configs.idx_end_oos_preds][columns_keys['Date']].strftime(
            "%Y-%m-%d") + ' 23:00'
        end_date = datetime.strptime(end_date, date_format)
        test_block_timestamps = pd.date_range(start=self.original_dataset.iloc[self.data_configs.idx_start_oos_preds]
        [columns_keys['Date']],
                                              end=end_date,
                                              freq='h')
        # Set datetime index to the dataframe
        results_df['Datetime'] = test_block_timestamps
        results_df.set_index(results_df['Datetime'], inplace=True)
        results_df.drop(columns=['Datetime'], inplace=True)

        # add target column
        df_target = self.original_dataset.filter(regex=features_keys['target']).iloc[
                    self.data_configs.idx_start_oos_preds:
                    self.data_configs.idx_end_oos_preds + 1]
        results_df[df_target.columns[0]] = df_target.values

        return results_df

    def get_exper_path(self):
        """
        returns the experiment path
        """
        return os.path.join(os.getcwd(), 'experiments', 'tasks', self.data_configs.task_name,
                            self.data_configs.exper_setup, self.model_configs['run_id'])

    def __save_results__(self, test_results_df):
        """
        Save recalibration results
        """
        exper_save_path = os.path.join(self.get_exper_path(), 'results')
        os.makedirs(exper_save_path, exist_ok=True)
        target_col_name = test_results_df.filter(regex=features_keys['target']).columns.tolist()[0]
        fn = target_col_name.replace(features_keys['target'], '')
        test_results_df.rename(columns={target_col_name: fn}, inplace=True)
        with open(exper_save_path + '/recalib_test_results-tuned-' + self.optuna_m + '.p', 'wb') as f:
            pickle.dump(test_results_df, f)

    def run_hyperparams_tuning(self, optuna_m: str = 'random', n_trials: int = 10):
        """
        Model hyperparameters tuning routine
        """

        def objective(trial):
            # Clear clutter from previous session graphs.
            tf.keras.backend.clear_session()
            # Update model configs with hyperparams trial
            self.model_configs = self.model_class.get_hyperparams_trial(trial=trial, settings=self.model_configs)

            # Build model using the current configs
            model = regression_model(settings=self.model_configs,
                                     sample_x=train_vali_block.x_vali[0:1])

            # Train model
            model.fit(train_x=train_vali_block.x_train, train_y=train_vali_block.y_train,
                      val_x=train_vali_block.x_vali, val_y=train_vali_block.y_vali,
                      pruning_call=TFKerasPruningCallback(trial, "val_loss"),
                      plot_history=False)

            # Compute val loss
            results = model.evaluate(x=train_vali_block.x_vali, y=train_vali_block.y_vali)
            return results

        # start from first train sample
        init_sample = 0
        # employ validation set till first test sample
        test_sample_idx = self.test_set_idxs[0]

        train_vali_block = self.__build_recalib_dataset_batches__(
            self.dataset[init_sample:test_sample_idx + self.data_configs.pred_horiz],
            fit_preproc=True).recalibBlocks[0]

        if optuna_m == 'grid_search':
            search_space = self.model_class.get_hyperparams_searchspace()
            sampler = optuna.samplers.GridSampler(search_space)
            pruner = None
        elif optuna_m == 'random':
            sampler = optuna.samplers.RandomSampler()
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)

        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        # Unique identifier of the study.
        study_name = (self.data_configs.task_name
                      + self.model_configs['model_class'] + '-'
                      + self.model_configs['PF_method']
                      + '-' + optuna_m)
        storage_name = "sqlite:///db.sqlite3"

        study = optuna.create_study(direction="minimize",
                                    sampler=sampler,
                                    pruner=pruner,
                                    storage=storage_name,  # Specify the storage URL here.
                                    study_name=study_name,
                                    load_if_exists=True
                                    )

        timeout = 3600 * 24.0 * 7  # 7 days
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print("Study statistics: ")

        print("Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            # store best hyper in the config dict
            self.model_configs[key] = value

        return self.model_class.get_hyperparams_dict_from_configs(self.model_configs)

    def get_model_hyperparams(self, method, optuna_m='random'):
        self.optuna_m = optuna_m
        self.hyper_mode = method
        path = os.path.join(self.get_exper_path(), 'tuned_hyperp-' + optuna_m + '.json')
        if method == 'load_tuned':
            print('-----------------------------------------')
            print('Loading tuned hyperparams')
            print('-----------------------------------------')
            with open(path) as f:
                return json.load(f)

        elif method == 'optuna_tuner':
            print('-----------------------------------------')
            print('Starting optuna tuner')
            model_hyperparams = self.run_hyperparams_tuning(optuna_m=optuna_m)
            print('-----------------------------------------')
            # save model hyperparams to json
            with open(path, 'w') as f:
                json.dump(model_hyperparams, f)
            return model_hyperparams
        else:
            sys.exit('ERROR: uknown hyperparam method')

    def run_recalibration(self, model_hyperparams: Dict, plot_history: object = False,
                          plot_weights: object = False) -> object:
        """
        Main recalibration loop
        """
        print('------------------------------------------------------------------------------')
        print('Starting recalibration of config: ' + str(self.model_configs['PF_method']))
        print('------------------------------------------------------------------------------')

        # List to store results over recalibration
        ensem_test_PIs = []

        # Iterate over test samples
        for i_t in range(0, self.test_set_idxs.shape[0]):
            tf.keras.backend.clear_session()
            print('Recalibrating test sample: ' + str(i_t + 1) + '/' + str(self.test_set_idxs.shape[0]))
            test_sample_idx = self.test_set_idxs[i_t]
            # Set index of first train sample, depending on the config

            init_sample = 0 if self.data_configs.keep_past_train_samples else i_t * self.data_configs.pred_horiz

            # Build the current recalibration batch including preprocessing (preprocess option)
            rec_samples = self.__build_recalib_dataset_batches__(
                self.dataset[init_sample:test_sample_idx + self.data_configs.pred_horiz],
                fit_preproc=True)

            # Get first rec_block in list
            rec_block = rec_samples.recalibBlocks[0]

            # Merge model configs and hyperparams tuning into the settings dict
            settings = {**self.model_configs, **model_hyperparams}

            # Create ensemble handler
            ensemble = Ensemble(settings=settings)

            # List to store ensemble components preds
            preds_test_e = []

            # Create and fit the ensemble components
            for e in range(settings['num_ense']):

                tf.keras.backend.clear_session()
                # Recalibrate every 90 days
                if i_t % 365 == 0:
                    model = regression_model(settings=settings,
                                             sample_x=rec_samples.x_test)

                    model.fit(train_x=rec_block.x_train, train_y=rec_block.y_train,
                              val_x=rec_block.x_vali, val_y=rec_block.y_vali,
                              plot_history=plot_history
                              )

                preds_test_e.append(model.predict(rec_samples.x_test))
                if plot_weights:
                    model.plot_weights()

            # Aggregate ensemble predictions

            ensem_preds_test = ensemble.aggregate_preds(preds_test_e)

            # Build and store the prediction quantiles for the current test samples using the selected method
            ens_p = ensemble.get_preds_test_quantiles(preds_test=ensem_preds_test)
            rescaled_PIs = {}
            for i in range(ens_p.shape[-1]):
                rescaled_PIs[self.model_configs['target_quantiles'][i]] = self.preproc['target'].inverse_transform(
                    ens_p[:, i:i + 1])[:, 0]
            results_df = pd.DataFrame(rescaled_PIs)

            if (self.GLM_regressor):
                for col in results_df.columns:
                    results_df[col] = (self.y_hat[i_t * self.data_configs.pred_horiz: ( i_t + 1) * self.data_configs.pred_horiz]- results_df[col]).values

            if (self.fourier_analysis):
                for col in results_df.columns:
                    results_df[col] = self.fourier_baseline.loc[
                                      test_sample_idx: test_sample_idx + self.data_configs.pred_horiz - 1].values + \
                                      results_df[col]

            # Rescale the holidays
            if self.holiday == 'Ratio':

                if self.original_dataset.loc[test_sample_idx: test_sample_idx + self.data_configs.pred_horiz - 1,
                   'CONST__Holiday'].values[0] == 1:

                    # Get the day of the week for the holiday
                    DoH = self.original_dataset.loc[test_sample_idx: test_sample_idx + self.data_configs.pred_horiz - 1,
                          'CONST__DoW'].values[0]

                    # Get data for the month before the holiday
                    month_before_holiday = self.original_dataset.loc[test_sample_idx - (30 * 24):test_sample_idx]
                    target_column = self.original_dataset.filter(regex='^TARG__').columns[0]
                    # Calculate the mean values for the normal day

                    mean_normal_day = month_before_holiday[month_before_holiday['CONST__DoW'] == DoH].groupby(
                        'Hour')[target_column].mean()

                    # Calculate the mean values for Sunday
                    mean_sunday_day = month_before_holiday[month_before_holiday['CONST__DoW'] == 0].groupby(
                        'Hour')[target_column].mean()

                    # Convert DataFrames to NumPy arrays
                    mean_normal_day_array = mean_normal_day.to_numpy()
                    mean_sunday_day_array = mean_sunday_day.to_numpy()

                    # Calculate the ratio
                    r = mean_sunday_day_array / mean_normal_day_array
                    r[:4] = 1

                    # Apply the ratio to each column in results_df
                    for col in results_df.columns:
                        results_df[col] = r * results_df[col]

            ensem_test_PIs.append(results_df)

        test_results_df = self.__transform_test_results__(pd.concat(ensem_test_PIs, axis=0))
        # plot_quantiles(test_results_df, target=features_keys['target']+self.data_configs.task_name)

        # Save results to file
        self.__save_results__(test_results_df)

        # Return test predictions
        return test_results_df
