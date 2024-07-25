''' Implementation of the GLM class'''

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import sys
from scipy import stats

class GLM():

    ''' Class which implements a linear regression
    using a One Hot Encoding scheme in order to reduce the
    trend and seasonality effect of the target variable '''

    def __init__(self, start_index: int):
        '''Constructor of the class'''

        # Inputs:
        # start_index:  start index of the dataframe

        self.start_index = start_index

    def compute_regressors(self, df: pd.DataFrame):
        ''' Method that builds the dataframe of the regression variables'''

        # Inputs:
        # df: dataframe from which to build the regressors dataframe

        # Outputs:
        # regressor_df: dataframe of the regressors

        # Reset the indexes
        df.reset_index(drop=True, inplace=True)

        # Compute the time index - we start counting from the first training index dataset
        t=(df.index+self.start_index)/(df.index.max() +self.start_index)
        regressor = pd.DataFrame(t, columns=['t'])

        # Generate 24 dummy variables to encode the hour of the day and concatenate with the regressor dataframe
        df['Hour'] = df['Hour'].astype('category')
        df_hour_dummies = pd.get_dummies(df['Hour'], prefix='Hour', drop_first=False).astype(int)
        df_all_hours=pd.DataFrame(0, index=df_hour_dummies.index, columns=['Hour_' + str(i) for i in range(24)])
        for hour in df_hour_dummies.columns:
            df_all_hours[hour] = df_hour_dummies[hour]

        regressor_df = pd.concat([regressor, df_all_hours], axis=1)

        # Generate 7 dummy variables to encode the day of the week and concatenate with the regressor dataframe
        df['CONST__DoW'] = df['CONST__DoW'].astype('category')
        df_week_dummies = pd.get_dummies(df['CONST__DoW'], prefix='Day of Week', drop_first=False).astype(int)
        df_all_days = pd.DataFrame(0, index=df_week_dummies.index, columns=['Day of Week_' + str(i) for i in range(7)])
        for day in df_week_dummies.columns:
            df_all_days[day] = df_week_dummies[day]
        regressor_df  = pd.concat([regressor_df , df_all_days], axis=1)

        # Generate 12 dummy variables to encode the month and concatenate with the regressor dataframe
        months = df['Date'].dt.month.astype('category')
        df_month_dummies = pd.get_dummies(months, prefix='Month', drop_first=False).astype(int)
        df_all_months = pd.DataFrame(0, index=df_month_dummies.index, columns=['Month_' + str(i) for i in range(1,13)])
        for month in df_month_dummies.columns:
            df_all_months[month] = df_month_dummies[month]
        regressor_df = pd.concat([regressor_df, df_all_months], axis=1)

        # Compute the interaction terms amongst hour-day of the week
        interaction=pd.DataFrame(index=regressor_df.index)
        for hour_var in df_all_hours.columns:
            for day_var in df_all_days.columns:
                interaction[f"Interaction {hour_var} x {day_var}"]=df_all_hours[hour_var] * df_all_days[day_var]
        # Add the interaction terms to the regressors dataset
        regressor_df = pd.concat([regressor_df, interaction], axis=1)

        # Compute the interaction terms amongst day of the week - month
        interaction = pd.DataFrame(index=regressor_df.index)
        for day_var in df_all_days.columns:
            for month_var in df_all_months.columns:
                interaction[f"Interaction {day_var} x {month_var}"] = df_all_days[day_var] * df_all_months[month_var]
        # Add the interaction terms to the regressors dataset
        regressor_df = pd.concat([regressor_df, interaction], axis=1)

        # Return
        return regressor_df


    def fit_GLM(self, training_dataset: pd.DataFrame, flag = "ElasticNet"):
        '''Method which builds and fits the linear model'''

        # Inputs:
        # training_dataset: dataset used for training
        # flag:             linear model to implement

        # Outputs:
        # self.residuals:   training residuals

        # Extract the training datasets
        self.training_dataset = training_dataset.copy()
        self.flag = flag
        training_features= self.compute_regressors(training_dataset)
        training_target=training_dataset['TARG__NetLoad']

        # Add the intercept
        X=sm.add_constant(training_features)

        if flag =="LASSO":
            # Initialize the LASSO model (we use LASSO due to the high number of variables)
            lasso = Lasso(random_state=42)  # Random to increase convergence

            # Use LASSO for regularization and variable selection
            param_grid = {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]
            }

            # Use 5-fold cross validation to choose the best lambda (running on all processors to speed it up)
            cv_lasso = RandomizedSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            cv_lasso.fit(X, training_target)
            self.fitted = cv_lasso.best_estimator_

            # print("Coefficients:", self.fitted.coef_)
            # print("Intercept:", self.fitted.intercept_)
            print(f"LASSO score (R-squared): {r2_score(training_target, self.fitted.predict(X))}")


        # Other option: elastic net
        elif flag == "ElasticNet":
            # Define the parameter grid for ElasticNet
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 10],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
            }

            # Use 5-fold cross validation to choose the best parameters (running on all processors to speed it up)
            elastic_net_cv = RandomizedSearchCV(ElasticNet(), param_grid, cv=5, scoring='neg_mean_squared_error',
                                                n_jobs=-1)
            elastic_net_cv.fit(X, training_target)
            self.fitted = elastic_net_cv.best_estimator_

            # Evaluate the model
            print(f"Elastic Net score (R-squared): {r2_score(training_target, self.fitted.predict(X))}")


        # Other option: OLS
        elif flag == "OLS":
            ols = sm.OLS(training_target, X)
            self.fitted = ols.fit()
            print(self.fitted.summary())

        else:
            sys.exit('ERROR: linear model not implemented')


        # Compute teh residuals
        self.residuals = self.fitted.predict(X)-training_target

        # Return
        return self.residuals

    def predict_GLM(self, testing_dataset: pd.DataFrame):
        '''Method which performs testing'''

        # Inputs:
        # testing_dataset: dataset used for testing

        # Outputs:
        # self.y_hat: predicted target variable
        # residuals:

        # Compute the regressor dataframe
        self.testing_dataset = testing_dataset
        testing_features= self.compute_regressors(testing_dataset)

        # Include the first const column for beta0
        const_column = testing_features.assign(const=1).pop('const')
        testing_features.insert(0, 'const', const_column)

        # Predict the values
        self.y_hat= self.fitted.predict(testing_features)

        # Compute the test residuals
        test_residuals = self.y_hat-testing_dataset['TARG__NetLoad']

        return self.y_hat, test_residuals

    def plot_residuals_GLM(self):
        '''Method which plots the training residuals of the model'''

        # Plot the residuals
        plt.plot(self.training_dataset.index, self.residuals, label='Residuals', linewidth=0.7, marker='.', markersize=2)
        plt.title('Residuals of ' + self.flag)
        plt.xlabel('Hour')
        plt.ylabel('Residuals')
        plt.show()

        # Plot autocorrelation function of the residuals
        plot_acf(self.residuals)
        plt.title("Autocorrelation of the residuals of of " + self.flag)
        plt.xlabel('Hour')
        plt.ylabel('Residuals')
        plt.show()

        # Plot partial autocorrelation function of the residuals
        plot_pacf(self.residuals)
        plt.title("Partial autocorrelation of the residuals of " + self.flag)
        plt.xlabel('Hour')
        plt.ylabel('Residuals')
        plt.show()

    def plot_prediction_GLM(self):
        '''Method which plots the true and predicted target values of the model'''

        # Plot the y and y_hat
        self.testing_dataset.reset_index(inplace=True, drop=True)
        plt.plot(self.testing_dataset.index, self.testing_dataset['TARG__NetLoad'], label='True load', linewidth=0.7)
        plt.plot(self.testing_dataset.index, self.y_hat, label='Predicted load of' + self.flag, linewidth=0.7)
        plt.legend(fontsize='medium', loc='upper right')

        # Get and plot the dates at the beginning of each year
        num_months = len(self.testing_dataset) // (24 * 365)
        month_indices = [i*24*7 for i in range(num_months)]
        labels = ["week {}".format(i) for i in month_indices]
        plt.xticks(month_indices, labels, rotation= 45)

        # Add labels and titles
        plt.xlabel('Dates')
        plt.ylabel('Load')
        plt.title(self.flag + ' prediction')
        plt.show()

    def residual_normality(self):
        '''Method that tests the normality assumption of the training residuals'''

        # Perform shapiro test
        result=stats.shapiro(self.residuals)
        print("-------------------------------------------------------------\n"
              "Shapiro test on residuals produced the following p-value: \n", result[1])
