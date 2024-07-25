''' Implementation of the VariableSelection class'''


import pandas as pd
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import statsmodels.api as sm

class VariableSelection():

    '''Class that performs PCA on FUTU or ALL variables to eliminate collinearity'''

    def __init__(self, dataset: pd.DataFrame, idx_start_oos_preds: int, data_configs: dict, preprocesser: dict, flag = "FUTU", LASSO_selection = False):
        '''Constructor of the class'''

        # Inputs:
        # dataset:               considered dataset
        # index_start_oos_preds: index with the start of the prediction
        # data_configs:          data configuration
        # preprocesser:          data preprocesser used before performing PCA
        # flag:                  if "FUTU" performs PCA only on FUTU variables
        #                        if "ALL" transforms hour/month/dayofyear through cos transformation
        #                        and performs PCA on all the variables
        # LASSO_selection:       if true: after performing PCA, LASSO performs variable selection on these
        #                        variables

        self.dataset= dataset
        self.idx_start_oos_preds = idx_start_oos_preds
        self.data_configs = data_configs
        self.preprocesser = preprocesser
        self.flag = flag
        self.LASSO_selection = LASSO_selection

    def preprocess_variables(self):
        '''Method that builds the variables on which PCA will be performed'''

        # Consider only the feature FUTU variables
        if self.flag == "FUTU":

            # Filter the dataset and assign these variables to the new dataset
            self.features = self.dataset.filter(regex='^FUTU')
            column_names = self.features.columns.tolist()

        elif self.flag == "ALL":
            # Create a new dataframe and store the numeric values (exploiting periodic
            # functions) of the seasonality and trend terms
            auxilary_dataset = pd.DataFrame()
            auxilary_dataset['Hour'] = np.cos(2 * np.pi * self.dataset['Hour'] / 24)
            auxilary_dataset['CONST__DoW'] = np.cos(2 * np.pi * self.dataset['CONST__DoW'] / 7)
            auxilary_dataset['CONST__DoY'] = np.cos(2 * np.pi * self.dataset['CONST__DoY'] / 365)
            auxilary_dataset['CONST__Month'] = np.cos(2 * np.pi * (self.dataset['Date'].dt.month - 1) / 12)
            auxilary_dataset['t'] = self.dataset.index / self.dataset.index.max()
            auxilary_dataset['CONST__Holiday'] = self.dataset['CONST__Holiday']

            # Substitute them in the new dataframe and eliminate the old ones
            const_columns = self.dataset.filter(regex='^CONST').columns.tolist()
            self.dataset.drop(const_columns, axis=1, inplace=True)

            idx_columns = self.dataset.filter(regex='^IDX').columns.tolist()
            self.dataset.drop(idx_columns, axis=1, inplace=True)

            self.dataset.drop('Hour', axis=1, inplace=True)
            self.dataset.drop('Date', axis=1, inplace=True)

            self.dataset = pd.concat([self.dataset, auxilary_dataset], axis=1)

            # Consider only the feature variables
            self.features = self.dataset.drop(self.dataset.filter(regex='^TARG'), axis=1)
            column_names = self.features.columns.tolist()

        else:
            sys.exit('ERROR: linear model not implemented')

        # Scale the variables
        self.preprocesser['feat'].fit(self.features)
        self.features = self.preprocesser['feat'].transform(self.features)
        self.features = pd.DataFrame(self.features, index=self.dataset.index, columns=column_names)

    def perform_PCA(self, feature_dataset: pd.DataFrame):
        '''Method that performs PCA'''

        # Inputs:
        # feature_dataset:        dataset on which to perform PCA

        # Perform the PCA on the preprocessed feature dataset
        pca = PCA(n_components=None)
        self.pca_dataset = pd.DataFrame(pca.fit_transform(feature_dataset))
        self.pca_dataset.columns = ["PC__" + str(i + 1) for i in range(self.pca_dataset.shape[1])]

        # Print PCA results
        print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative Explained Variance: {np.cumsum(pca.explained_variance_ratio_)}")


    def LASSO (self, dataset: pd.DataFrame, beta_threshold = 100):
        '''Method that implements LASSO'''

        # Inputs:
        # dataset:          dataset containing the regressor variables and the target variables

        # Outputs:
        # significant_features:     list of strings containing the significant column names

        # Consider the training dataset
        data = dataset[: self.idx_start_oos_preds]

        # Retrieve features and target variables
        feature_train = data.drop(self.dataset.filter(regex='^TARG'), axis=1)
        target_train = data.filter(regex='^TARG')


        # Train the Lasso model
        #lasso_model = LassoCV(cv=5, random_state=42)
        #lasso_model = Lasso(alpha=0.01, random_state=42)
        #lasso_model.fit(feature_train, target_train)

        model = sm.OLS(target_train, feature_train)
        result =model.fit_regularized(method='elastic_net', alpha=0.0, L1_wt=0.1)

        # Plot the betas
        betas = result.params

        # Plot betas
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(betas)), betas, tick_label=["PC__{}".format(i) for i in range(1,len(betas)+1)])
        plt.xlabel('Beta Values')
        plt.title('Beta Values for Principal Components')
        plt.show()

        # Filter features based on the beta threshold
        significant_features = feature_train.columns[np.abs(betas)> beta_threshold]

        return significant_features

    def selection(self):
        '''Method that goes through all the procedure'''

        # Outputs:
        # selected_features_dataset:   dataset after having performed PCA. It contains all PC__ variables if
        #                              self.flag=="ALL", PC__, CONST__, IDX__ variables if self.flag=="FUTU__"
        #                              where PCA was performed only on "FUTU__" variables

        # Build the dataframe on which to perform PCA
        self.preprocess_variables()

        # Perform the PCA on the preprocessed variables
        self.perform_PCA(self.features)

        if self.LASSO_selection:
            # Perform the LASS0 and save the significant features
            dataset=pd.concat([self.pca_dataset, self.dataset.filter(regex='^TARG__')], axis=1)
            significant_features = self.LASSO(dataset)

        else:
            # The significant features are all the PC__ variables
            significant_features = self.pca_dataset.filter(regex='^PC__').columns


        # Build the dataset with the selected features
        selected_features = pd.concat([self.pca_dataset[significant_features].reset_index(drop=True), self.dataset.filter(regex='^IDX').reset_index(drop=True),
                                       self.dataset.filter(regex='^CONST').reset_index(drop=True)], axis=1) if self.flag == "FUTU" else \
            self.pca_dataset[significant_features].reset_index(drop=True)

        # Add the target variable
        selected_features_dataset = pd.concat([selected_features, self.dataset.filter(regex='^TARG__')], axis=1)

        # Look at the correlation
        correlation_matrix = selected_features_dataset.corr()

        # Return
        return selected_features_dataset
