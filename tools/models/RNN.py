from tools.data_utils import features_keys
import numpy as np
import tensorflow as tf
from typing import List
import sys
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import os
import shutil
import matplotlib.pyplot as plt


class RNNRegressor:
    '''Class which implements the RNN model '''

    def __init__(self, settings, loss):
        '''Constructor of the class'''

        # Inputs:
        # settings:     configuration settings
        # loss:         loss function to be used

        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):
        '''Method that builds the model architecture'''

        # Inputs:
        # loss:         loss function to be used

        # Outputs:
        # self.model:   compiled model

        # Values flow through here
        x_in = tf.keras.layers.Input(shape=(self.settings['n_time_steps'], self.settings['n_features']), )
        x_in = tf.keras.layers.BatchNormalization()(x_in)

        # Add LSTM layer
        x = tf.keras.layers.LSTM(self.settings['LSTM_size'],
                                 activation=self.settings['activation'],
                                 return_sequences=(self.settings['n_hidden_layers'] > 1),
                                 dropout=self.settings['dropout'],
                                 recurrent_dropout = self.settings['recurrent dropout'])(x_in)

        # Add the hidden LSTM layers
        for hl in range(self.settings['n_hidden_layers'] - 1):
            return_sequences = (hl < self.settings['n_hidden_layers'] - 2)
            x = tf.keras.layers.LSTM(self.settings['LSTM_size'],
                                     activation=self.settings['activation'],
                                     return_sequences=return_sequences,
                                     dropout=self.settings['dropout'],
                                     recurrent_dropout=self.settings['recurrent dropout']
                                    )(x)

        # Output layer
        if self.settings['PF_method'] == 'point':
            out_size = 1
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear')(x)
            output = tf.keras.layers.Reshape((self.settings['pred_horiz'], 1))(logit)

        elif self.settings['PF_method'] == 'qr':
            out_size = len(self.settings['target_quantiles'])
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size, activation='linear')(x)
            output = tf.keras.layers.Reshape((self.settings['pred_horiz'], out_size))(logit)
            output = tf.keras.layers.Lambda(lambda x: tf.sort(x, axis=-1))(output)

        elif self.settings['PF_method'] == 'Normal':
            out_size = 2
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size, activation='linear')(x)
            output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :self.settings['pred_horiz']],
                                                                        scale=1e-3 + 3 * tf.math.softplus(0.05 * t[...,
                                                                                                                 :
                                                                                                                 self.settings[
                                                                                                                     'pred_horiz']:])))(
                logit)

        elif self.settings['PF_method'] == 'JSU':
            out_size = 4
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size, activation='linear')(x)
            output = tfp.layers.DistributionLambda(lambda t: tfd.JohnsonSU(
                loc=t[..., :self.settings['pred_horiz']],
                scale=1e-3 + 3 * tf.math.softplus(
                    0.05 * t[..., self.settings['pred_horiz']:2 * self.settings['pred_horiz']]),
                skewness=t[..., 2 * self.settings['pred_horiz']:3 * self.settings['pred_horiz']],
                tailweight=1 + 3 * tf.math.softplus(
                    t[..., 3 * self.settings['pred_horiz']:4 * self.settings['pred_horiz']])
            ))(logit)

        else:
            sys.exit('ERROR: unknown PF_method config!')

        self.model = tf.keras.Model(inputs=[x_in], outputs=[output])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']),
                           loss=loss)
        self.model.summary()

    def fit(self, train_x, train_y, val_x, val_y, verbose=1, pruning_call=None):
        '''Method that fits the model'''

        # Inputs:
        # train_x:      training features
        # train_y:      training targets
        # val_x:        validation features
        # val_y:        validation targets
        # verbose:      verbosity level
        # pruning_call: pruning callback

        # Outputs:
        # history: training history

        # Prepare training and validation inputs
        train_x = self.build_model_input_from_series(x=train_x,
                                                     col_names=self.settings['x_columns_names'],
                                                     pred_horiz=self.settings['pred_horiz'])

        val_x = self.build_model_input_from_series(x=val_x,
                                                   col_names=self.settings['x_columns_names'],
                                                   pred_horiz=self.settings['pred_horiz'])
        # Early stopping callback
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=self.settings['patience'],
                                              restore_best_weights=False)

        # Checkpoint callback
        checkpoint_path = os.path.join(os.getcwd(), 'tmp_checkpoints', 'cp.weights.h5')
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor="val_loss", mode="min",
                                                save_best_only=True,
                                                save_weights_only=True, verbose=0)
        callbacks = [es, cp] if pruning_call is None else [es, cp, pruning_call]

        # Train the model
        history = self.model.fit(train_x,
                                 train_y,
                                 validation_data=(val_x, val_y),
                                 epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'],
                                 callbacks=callbacks,
                                 verbose=verbose)
        self.model.load_weights(checkpoint_path)
        shutil.rmtree(checkpoint_dir)
        return history

    def predict(self, x):
        '''Method to predict on testing data'''

        # Inputs:
        # x:        input features for prediction

        # Outputs:
        # predictions:  model predictions

        x = self.build_model_input_from_series(x=x,
                                               col_names=self.settings['x_columns_names'],
                                               pred_horiz=self.settings['pred_horiz'])
        return self.model(x)

    def evaluate(self, x, y):
        '''Method that evaluates the model on data'''

        # Inputs:
        # x:        input features for evaluation
        # y:        true target values for evaluation

        x = self.build_model_input_from_series(x=x,
                                               col_names=self.settings['x_columns_names'],
                                               pred_horiz=self.settings['pred_horiz'])
        return self.model.evaluate(x=x, y=y)

    @staticmethod
    def build_model_input_from_series(x, col_names: List, pred_horiz: int):
        '''Method that prepares model inputs'''

        # Inputs:
        # x:            input data
        # col_names:    column names of the input data
        # pred_horiz:   prediction horizon

        # Outputs:
        # in1: prepared past feature inputs
        # in2: prepared future feature inputs

        # Get index of target and past features
        past_col_idxs = [index for index, item in enumerate(col_names) if
                         features_keys['target'] in item or features_keys['past'] in item]

        # Get index of timekeeping features
        time_col_idxs = [index for index, item in enumerate(col_names) if features_keys['const'] in item]

        # Get index of dummy features
        dummy_col_idxs = [index for index, item in enumerate(col_names) if features_keys['dummy'] in item]

        # Get index of future features
        futu_col_idxs = [index for index, item in enumerate(col_names) if features_keys['futu'] in item]

        # Get indexes of the modified PC variables
        pc_col_idxs = [index for index, item in enumerate(col_names) if features_keys['pc'] in item]

        # Build conditioning variables for past features
        past_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in past_col_idxs]

        # Build conditioning variables for dummy features defined in PrTSF_Recalib_tools
        dummy_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in dummy_col_idxs]

        # Build conditioning variables for features constant during the day
        c_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in time_col_idxs if
                  'CONST__Hour' not in col_names[feat_idx] and 'CONST__Holiday' not in col_names[feat_idx]]

        # Extract hours of next day feature if present
        hour_feat = None
        for index, item in enumerate(col_names):
            if features_keys['hour'] in item:
                hour_feat = [x[:, :-pred_horiz, index]]
                break

        # Build conditioning variables for future features
        futu_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in futu_col_idxs]

        # Calculate min and max for weather forcast features in 24-step blocks
        min_max_futu = []
        for feat_array in futu_feat:
            reshaped_array = feat_array.reshape(feat_array.shape[0], -1, 24)
            min_vals = np.min(reshaped_array, axis=2)
            max_vals = np.max(reshaped_array, axis=2)
            min_max_futu.append(min_vals)
            min_max_futu.append(max_vals)

        # Build conditioning variables for transformed weather forcast features
        pc_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in pc_col_idxs]

        # Convert and transpose features
        if c_feat:
            converted_feat_hour = tf.convert_to_tensor(hour_feat)
            transposed_feat_hour = tf.transpose(converted_feat_hour, perm=[1, 0, 2])
        if pc_feat:
            converted_feat_pc = tf.convert_to_tensor(pc_feat)
            transposed_feat_pc = tf.transpose(converted_feat_pc, perm=[1, 0, 2])
        if c_feat:
            converted_feat_c = tf.convert_to_tensor(c_feat)
            transposed_feat_c = tf.transpose(converted_feat_c, perm=[1, 0, 2])
        if dummy_feat:
            converted_feat_dummy = tf.convert_to_tensor(dummy_feat)
            transposed_feat_dummy = tf.transpose(converted_feat_dummy, perm=[1, 0, 2])
        if futu_feat:
            converted_feat_futu = tf.convert_to_tensor(futu_feat)
            transposed_feat_futu = tf.transpose(converted_feat_futu, perm=[1, 0, 2])

        # Transpose
        converted_feat_past = tf.convert_to_tensor(past_feat)
        transposed_feat_past = tf.transpose(converted_feat_past, perm=[1, 0, 2])


        if c_feat and futu_feat:
            # Concatenate future, constant, and hour features
            input = tf.concat([transposed_feat_c, transposed_feat_futu, transposed_feat_past,transposed_feat_hour], axis=1)

        elif c_feat and pc_feat:
            # Concatenate transformed future and dummy features
            input = tf.concat([transposed_feat_c, transposed_feat_pc,transposed_feat_hour], axis=1)


        elif dummy_feat and futu_feat:
            # Concatenate future  and dummy features
            input = tf.concat([transposed_feat_futu, transposed_feat_dummy, transposed_feat_past ], axis=1)

        elif dummy_feat and pc_feat:
            input = tf.concat([transposed_feat_pc, transposed_feat_dummy], axis=1)

        return input

    @staticmethod
    def get_hyperparams_trial(trial, settings):
        '''Method that defines hyperparameter search space and suggest values for trials'''

        # Inputs:
        # trial:        object used to suggest hyperparameter values
        # settings:     dictionary that stores the suggested hyperparameter values

        # Outputs:
        # settings: updated dictionary with hyperparameters

        settings['LSTM_size'] = trial.suggest_int('LSTM_size', 24, 214, step=24)
        settings['n_hidden_layers'] = trial.suggest_int('n_hidden_layers',1,4)
        settings['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        settings['activation'] = trial.suggest_categorical('activation', ['tanh', 'sigmoid', 'relu', 'leaky_relu'])
        settings['dropout'] = trial.suggest_float('dropout',0,0.6,step=0.1)
        settings['recurrent dropout'] = trial.suggest_float('recurrent dropout', 0, 0.4, step=0.1)
        return settings

    @staticmethod
    def get_hyperparams_searchspace():
        '''Method that defines the search space for hyperparameter optimization'''

        # Outputs:
        # Dictionary defining the search space for hyperparameters

        return {'LSTM_size': [24, 512],
                'lr': [1e-5, 1e-3],
                'n_hidden_layers': [1, 4],
                'activation': ['tanh', 'sigmoid', 'relu', 'leaky_relu'],
                'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                'recurrent dropout': [ 0, 0.1, 0.2, 0.3, 0.4]}

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):
        '''Method that extracts model hyperparameters'''

        # Inputs:
        # configs:      dictionary with hyperparameter values

        # Outputs:
        # model_hyperparams:    dictionary with model hyperparameters extracted

        model_hyperparams = {
            'hidden_size': configs['hidden_size'],
            'n_hidden_layers': configs['n_hidden_layers'],
            'lr': configs['lr'],
            'activation': configs['activation'],
            'dropout': configs['dropout'],
            'recurrent_dropout': configs['recurrent_dropout']
        }
        return model_hyperparams

    def plot_weights(self):
        '''Method to plot the weights'''

        lstm_layer = self.model.layers[1]
        w_b = lstm_layer.get_weights()
        plt.imshow(w_b[0], aspect='auto')
        plt.title('RNN input weights')
        plt.show()
