

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


class CNNMIXEDRegressor:
    '''Class which implements the CNN mixed model '''

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

        # Past values flow through here
        x_in_1 = tf.keras.layers.Input(shape=(self.settings['n_time_steps'], self.settings['n_features']), )
        x_in_1 = tf.keras.layers.BatchNormalization()(x_in_1)
        x = x_in_1

        # Add Conv1D layers
        for hl in range(self.settings['n_encoder_layers']):
            x = tf.keras.layers.Conv1D(self.settings["filter_size"], self.settings["kernel_size"], activation='relu',
                                       input_shape=(self.settings['n_time_steps'], self.settings['n_features']), padding = 'same')(x)

        # Reshape the data for LSTM layers
        x = tf.keras.layers.Reshape((self.settings['filter_size'], self.settings['n_time_steps']))(x)

        # Add LSTM layers
        for hl in range(self.settings['n_LSTM_layers']):

            return_sequences = True
            x = tf.keras.layers.LSTM(self.settings['LSTM_size'],
                                     activation=self.settings['LSTM_activation'],
                                     return_sequences=return_sequences,  #LSTM outputs a sequence of vectors of length equal to the input sequence length.
                                     implementation=2,
                                     dropout=self.settings['dropout'],
                                     recurrent_dropout = self.settings['recurrent dropout'],
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)

        x_out_1 = tf.math.reduce_mean(x, axis=1, keepdims=False, name=None)

        # Future values flow through here
        x_in_2 = tf.keras.layers.Input(shape=(self.settings['dense_input_size']), )
        x_in_2 = tf.keras.layers.BatchNormalization()(x_in_2)
        z = x_in_2

        # Add Dense layers
        for hl in range(self.settings['n_dense_layers'] - 1):
            z = tf.keras.layers.Dense(self.settings['dense_size'],
                                      activation=self.settings['DENSE_activation'],
                                      )(z)
        x_out_2 = tf.keras.layers.Dense(self.settings['dense2_final_size'], activation='linear')(z)
        x_out_2 = tf.keras.layers.Flatten()(x_out_2)

        # Concatenate to last dense steps
        x = tf.keras.layers.Concatenate(axis=-1)([x_out_1, x_out_2])

        # Add final Dense layers
        for hl in range(self.settings['n_final_dense_layers'] - 1):
            x = tf.keras.layers.Dense(self.settings['final_dense_size'],
                                      activation=self.settings['DENSE_activation'],
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

        # Compile the model
        self.model = tf.keras.Model(inputs=[x_in_1, x_in_2], outputs=[output])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']),
                           loss=loss)
        self.model.summary()

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
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

        train_x1, train_x2 = self.build_model_input_from_series(x=train_x,
                                                                col_names=self.settings['x_columns_names'],
                                                                pred_horiz=self.settings['pred_horiz'])

        val_x1, val_x2 = self.build_model_input_from_series(x=val_x,
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
        history = self.model.fit({"input_3": train_x1, "input_4": train_x2},
                                 train_y,
                                 validation_data=([val_x1, val_x2], val_y),
                                 epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'],
                                 callbacks=callbacks,
                                 verbose=verbose)
        if os.path.exists(checkpoint_path):
            self.model.load_weights(checkpoint_path)
        shutil.rmtree(checkpoint_dir)
        return history

    def predict(self, x):
        '''Method to predict on testing data'''

        # Inputs:
        # x:        input features for prediction

        # Outputs:
        # predictions:  model predictions

        x1, x2 = self.build_model_input_from_series(x=x,
                                                    col_names=self.settings['x_columns_names'],
                                                    pred_horiz=self.settings['pred_horiz'])
        return self.model({"input_3": x1, "input_4": x2})

    def evaluate(self, x, y):
        '''Method that evaluates the model on data'''

        # Inputs:
        # x:        input features for evaluation
        # y:        true target values for evaluation

        x1, x2 = self.build_model_input_from_series(x=x,
                                                    col_names=self.settings['x_columns_names'],
                                                    pred_horiz=self.settings['pred_horiz'])
        return self.model.evaluate(x=[x1, x2], y=y)

    @staticmethod
    def build_model_input_from_series(x, col_names: List[str], pred_horiz: int):
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
        dummy_feat = [x[:, :, feat_idx] for feat_idx in dummy_col_idxs]


        # Build conditioning variables for features constant during the day
        c_feat = [x[:, -pred_horiz:-pred_horiz + 1, feat_idx] for feat_idx in time_col_idxs if
                  'CONST__Hour' not in col_names[feat_idx] and 'CONST__Holiday' not in col_names[feat_idx]]

        # Extract hours of next day feature if present
        hour_feat = None
        for index, item in enumerate(col_names):
            if features_keys['hour'] in item:
                hour_feat = [x[:, -pred_horiz:, index]]
                break

        # Build conditioning variables for future features
        futu_feat = [x[:, -pred_horiz:, feat_idx] for feat_idx in futu_col_idxs]

        # Calculate min and max for weather forcast features in 24-step blocks
        min_max_futu = []
        for feat_array in futu_feat:
            reshaped_array = feat_array.reshape(feat_array.shape[0], -1, 24)
            min_vals = np.min(reshaped_array, axis=2)
            max_vals = np.max(reshaped_array, axis=2)
            min_max_futu.append(min_vals)
            min_max_futu.append(max_vals)

        # Build conditioning variables for transformed weather forcast features
        pc_feat = [x[:, -pred_horiz:, feat_idx] for feat_idx in pc_col_idxs]

        # Convert and transpose past features
        converted_feat_past = tf.convert_to_tensor(past_feat)

        # Transpose the past features
        transposed_feat_past = tf.transpose(converted_feat_past, perm=[1, 2, 0])
        in1 = transposed_feat_past

        if c_feat and futu_feat:
            # Concatenate future, constant, and hour features
            in2 = np.concatenate(futu_feat + c_feat + hour_feat, axis=1)

        elif c_feat and pc_feat:
            # Concatenate transformed future and dummy features
            in2 = np.concatenate(pc_feat + c_feat + hour_feat, axis=1)


        elif dummy_feat and futu_feat:
            # Concatenate future  and dummy features
            in2 = np.concatenate(futu_feat + dummy_feat, axis=1)

        elif dummy_feat and pc_feat:
            in2 = np.concatenate(pc_feat + dummy_feat, axis=1)

        return in1, in2

    @staticmethod
    def get_hyperparams_trial(trial, settings):
        '''Method that defines hyperparameter search space and suggest values for trials'''

        # Inputs:
        # trial:        object used to suggest hyperparameter values
        # settings:     dictionary that stores the suggested hyperparameter values

        # Outputs:
        # settings: updated dictionary with hyperparameters

        settings['LSTM_size'] = 168
        settings['n_dense_layers'] = trial.suggest_int('n_dense_layers', 1, 4)
        settings['lr'] = trial.suggest_float('lr', 1e-5, 1e-3 ,  log=True)
        settings['n_encoder_layers'] = trial.suggest_int('n_encoder_layers', 1, 2)
        settings['dropout'] = trial.suggest_float('dropout', 0.1 , 0.9, step = 0.1 )
        settings['recurrent dropout'] = trial.suggest_float('recurrent dropout', 0.1 , 0.6, step=0.1 )
        settings['dense_size'] = trial.suggest_int('dense_size', 64, 512, step=50)
        settings['dense2_final_size'] = trial.suggest_int('dense2_final_size', 64, 512, step=50)
        settings['dense1_final_size'] = trial.suggest_int('dense1_final_size', 64, 512, step=50)
        settings['n_final_dense_layers'] = trial.suggest_int('n_final_dense_layers', 1, 3)
        settings['final_dense_size'] = trial.suggest_int('final_dense_size', 64, 512, step=50)
        settings['kernel_size'] = trial.suggest_int('kernel_size', 1, 3)
        settings['filter_size'] = trial.suggest_int('filter_size', 1, 10)
        settings['n_LSTM_layers'] = trial.suggest_int('n_LSTM_layers', 1, 3)
        settings['LSTM_activation'] = trial.suggest_categorical('LSTM_activation', ['relu', 'tanh', 'sigmoid', 'softmax'])
        settings['DENSE_activation'] = trial.suggest_categorical('DENSE_activation', ['relu', 'tanh', 'sigmoid', 'softmax', 'elu'])
        return settings


    @staticmethod
    def get_hyperparams_searchspace():
        '''Method that defines the search space for hyperparameter optimization'''

        # Outputs:
        # Dictionary defining the search space for hyperparameters

        return {"LSTM_size": [168], "n_LSTM_layers": [1, 3], "lr": [0.0001, 0.001], "n_encoder_layers": [1, 2],
                "n_dense_layers": [1, 4], "dense_size": [24, 512], "dense2_final_size": [24, 512],
                "dense1_final_size": [24, 512],"DENSE_activation": ["relu", "tanh", "sigmoid", "softmax", "elu"],
                "n_final_dense_layers": [1, 3], "final_dense_size": [24, 512], "filter_size": [1,10],
                "LSTM_activation": ["relu", "tanh", "sigmoid", "softmax"], 'kernel_size': [1,10],
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9],
                'recurrent dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                }

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):
        '''Method that extracts model hyperparameters'''

        # Inputs:
        # configs:      dictionary with hyperparameter values

        # Outputs:
        # model_hyperparams:    dictionary with model hyperparameters extracted

        model_hyperparams = {
            'hidden_size': configs['LSTM_size'],
            'n_hidden_layers': configs['n_LSTM_layers'],
            'lr': configs['lr'],
            'n_encoder_layers': configs['n_encoder_layers'],
            'n_dense_layers': configs['n_dense_layers'],
            'dense_size': configs['dense_size'],
            'dense2_final_size': configs['dense2_final_size'],
            'dense1_final_size': configs['dense1_final_size'],
            'n_final_dense_layers': configs['n_final_dense_layers'],
            'final_dense_size': configs['final_dense_size'],
            'kernel_size': configs['kernel_size'],
            'LSTM_activation': configs['LSTM_activation'],
            'DENSE_activation': configs['DENSE_activation'],
            'dropout': configs['dropout'],
            'recurrent dropout': configs['recurrent dropout'],
            'filter_size': configs['filter_size']
        }
        return model_hyperparams

    def plot_weights(self):
        '''Method to plot the weights'''

        lstm_layer = self.model.layers[1]
        w_b = lstm_layer.get_weights()
        plt.imshow(w_b[0], aspect='auto')
        plt.title('RNN input weights')
        plt.show()
