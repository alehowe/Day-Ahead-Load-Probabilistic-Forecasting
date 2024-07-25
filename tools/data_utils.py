"""
Definition of the keys employed by the data management utils.
Plots of the time series and fourier analysis
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from scipy import fft
import scipy.signal as sig
import math
from sklearn.linear_model import LinearRegression
import numpy as np
from itertools import islice


columns_keys = {
    'Date': 'Date',
    'Hour': 'Hour',
    'idx_global': 'IDX__global',
    'idx_step': 'IDX__step',
    'idx_sub_step': 'IDX__sub_step'
}

#--------------------------------------------------------------------------------
# Features naming conventions employed to build the samples by moving windows
#--------------------------------------------------------------------------------
features_keys={

    'holiday':'CONST__Holiday',

    'hour': 'CONST__Hour',
    # Employ just a single value related to the prediction step
    'const': 'CONST__',
    # Employ just a single value related to the IDX variables
    'idx': 'IDX__',
    # Target variable
    'target': 'TARG__',
    # Employ just values included in the configured moving window till the step before predictions (i.e., lags)
    'past': 'PAST__',
    # Employ the series value related to the prediction step and eventual substeps (in case of multi-step predictions)
    'futu': 'FUTU__',
    # True target variable
    'true': 'TRUE__',
    # Transformed variables
    'pc': 'PC__',
    # One hot encoded dates (HoD, DoW,MoY)
    'dummy': 'DUMMY__'
}


def get_dataset_save_path():
    return os.path.join(os.getcwd(), 'data', 'datasets')


class data_plotting():

    '''Class used for plotting the data'''

    def __init__(self, ds: pd.core.frame.DataFrame, configs: dict,  initial_date=datetime(year=2015, month=1, day=1), last_date=datetime(year=2017, month=12, day=31)):
        ''' Constructor of the class'''

        # Inputs:
        # ds:               dataset used for plotting
        # configs:          configs
        # initial_date:     initial of the plotting
        # last_date:        last date of the plotting

        ds['Date'] = pd.to_datetime(ds['Date'])
        self.plotting_df=ds[ (ds['Date'] >= initial_date) & (ds['Date'] <= last_date) ]
        self.target=['TARG__NetLoad' if configs['data_config'].task_name == 'NetLoad' else sys.exit('Plotting data has not been defined for Price')]
        self.start_date=initial_date
        self.end_date=last_date

    def plot_overall(self):
        """ Method that plots the sum target trend during each day across the considered period"""

        # Compute the dataset taking the sum of the target variable for each day
        df_sum_day = self.plotting_df.groupby('Date').agg({
            'CONST__DoW': 'first',
            'CONST__DoY': 'first',
            'CONST__Holiday': 'first',
            'TARG__NetLoad': 'sum'
        }).reset_index()

        plt.plot(df_sum_day['Date'], df_sum_day['TARG__NetLoad'], label='Sum of Load by day', linewidth=0.7)

        # Highlight the end of each week
        plt.plot(df_sum_day[df_sum_day['CONST__DoW'] == 0]['Date'], df_sum_day[df_sum_day['CONST__DoW'] == 0]['TARG__NetLoad'],
         linestyle='', color='blue', marker='.', markersize=4, label='Sunday')

        # Highlight holiday dates
        plt.plot(df_sum_day[df_sum_day['CONST__Holiday'] == 1]['Date'],
                 df_sum_day[df_sum_day['CONST__Holiday'] == 1]['TARG__NetLoad'],
                 linestyle='', color='red', marker='.', markersize=4, label='Holiday')

        plt.legend(fontsize='medium', loc='upper right')

        # Get and plot the dates at the beginning of each year
        plt.xticks(df_sum_day['Date'][(df_sum_day['Date'].dt.day==1) & (df_sum_day['Date'].dt.month==1) ])

        plt.xlabel('Day')
        plt.ylabel('Load')
        plt.title('Sum of Load by day')
        plt.show()

    def plot_month_hour_seasonality(self, month_number: int):
        """ Method that plots the target trend during the given month of different years"""

        # Inputs:
        # month_number:     month to plot (in integers 1-12)

        # Select the dataset and plot the desired hours
        days=[]
        for year in range(self.start_date.year, self.end_date.year+1):

            # Select the data at the given month and year
            df_month = self.plotting_df[(self.plotting_df['Date'].dt.month == month_number) & (self.plotting_df['Date'].dt.year == year)]
            df_month.reset_index(drop=True, inplace=True)

            # If the first of January of the first year shift by 1 day (since data is available from 2nd January)
            if (month_number == 1 and int(year)==2014):
                df_month.index = df_month.index + 24

            # Plot the month data of the given year
            plt.plot(df_month.index, df_month['TARG__NetLoad'], label=f'Year {year}', linewidth=0.7)
            days=df_month.index // 24 + 1

        # Days to highlight on the x-axis(one every 7 days)
        looking_days=[1,8,15,22,29]
        highlighted_indices = []

        # Iterate through looking_days and find the first index in days that matches each looking_day
        for day in looking_days:
            index = next(i for i, d in enumerate(days) if d == day)
            highlighted_indices.append(index)

        # Set the x-ticks
        plt.xticks(highlighted_indices, ["Day " + str(day) for day in looking_days])

        # Set the labels and the title
        plt.title(f'{calendar.month_name[month_number]} load Trends')
        plt.xlabel('Date')
        plt.ylabel('Net Load')
        plt.legend()
        plt.show()

    def plot_weekday_weekend_hour_seasonality(self, month_number: int, year: int):
        """ Method that plots the target trend over 24 hours of the first week (starting from Monday) of the given month and year
            of the given month and year"""

        # Inputs:
        # month_number:     month considered
        # year:             year considered

        # Consider the portion of the dataset of the desired month
        df_month=self.plotting_df[(self.plotting_df['Date'].dt.month==month_number) & (self.plotting_df['Date'].dt.year==year)].reset_index(drop=True)

        # Find the first index corresponding to the first Monday of the month
        first_monday = next(i for i, d in enumerate(df_month['CONST__DoW']) if d == 1)

        # Find the last index corresponding to the Sunday of the week
        next_monday = first_monday + 7*24

        # Consider the dataset in these days
        df_month=df_month[(df_month.index>=first_monday) & (df_month.index<next_monday)]

        # For each day plot the Load for each hour
        colours = sns.color_palette()

        for day_of_week in range(0,7):
            df_day= df_month[df_month['CONST__DoW'] == day_of_week]

            plt.plot(df_day['Hour'], df_day['TARG__NetLoad'], label=calendar.day_name[(day_of_week+6)%7], color=colours[(day_of_week+6)%7], linewidth=0.7)

        # Get and plot the hours and display the labels
        plt.legend(fontsize='medium', loc='upper right')
        plt.xticks(df_month['Hour'])
        plt.xlabel('Hour')
        plt.ylabel('Load')
        plt.title(f'Load per hour for the first week of {calendar.month_name[month_number]}')
        plt.show()

    def plot_monthly_netload_data(self, months: list):
        """Method that plots the target data for different months during the first year 2015"""

        # Inputs:
        # months:       list of months considered in the plot

        # Initialize a dictionary to store netload data for each month
        netload_data = {}

        # Extract data for each month
        for month in months:
            month_data = self.plotting_df[self.plotting_df['Date'].dt.strftime('%Y-%m') == f'2015-{month}']
            netload_data[month] = month_data['TARG__NetLoad'].values

        plt.figure(figsize=(15, 6))
        # Define colors for each month
        colors = ['blue', 'orange', 'green', 'red']

        for i, month in enumerate(months):
            x = range(len(netload_data[month]))
            # Shift x by 24 hours for January data 2014
            if month == '01':
                x = [val + 24 for val in x]
            plt.plot(x, netload_data[month], color=colors[i], label=f'Month {month}')

        # Add labels and legend
        plt.xlabel('Hours')
        plt.ylabel('TARG__NetLoad')
        plt.title('Netload Data for Different Months')
        plt.legend()
        plt.show()

    def plot_monthly_heatmap(self, months: list):
        """Plots the netload data for different months as heatmaps during the first year 2014"""

        # Inputs:
        # months:       list of months considered in the plot

        rainbow_colors = [(0, 'blue'), (0.25, 'cyan'), (0.5, 'green'), (0.75, 'yellow'), (1, 'red')]
        cmap_rainbow = LinearSegmentedColormap.from_list('rainbow', rainbow_colors)

        # Create subplots and plot
        fig, axs = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1, 1, 1]})

        for i, month in enumerate(months):
            # Find the given month
            month_data = self.plotting_df[self.plotting_df['Date'].dt.strftime('%Y-%m') == f'2015-{month}']

            # Heatmap for the current subplot
            heatmap_data = month_data.pivot_table(index=month_data['Date'].dt.day, columns='Hour',
                                                  values='TARG__NetLoad',
                                                  aggfunc=np.mean)  # useless

            sns.heatmap(heatmap_data, cmap=cmap_rainbow, annot=False, fmt=".1f", ax=axs[i])
            axs[i].set_title(f'{month}/2014')
            axs[i].set_xlabel('Hours')
            axs[i].set_ylabel('Days')

        plt.tight_layout()
        plt.show()

    def plot_netload_days(self, months: list, days: list):
        """Plots the netload data for specific days of each month in separate subplots"""

        # Inputs:
        # months:       list of months considered in the plot
        # days:         days plotted for each month

        # Create subplots
        fig, axs = plt.subplots(len(months), 1, figsize=(15, 6 * len(months)))
        global_min = float('inf')
        global_max = float('-inf')
        if len(months) == 1:
            axs = [axs]

        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']

        for idx, month in enumerate(months):
            # Initialize a dictionary to store netload data for each day
            netload_data = {}

            # Extract data for each day in the specified month
            for day in days:
                day_str = f'2015-{month}-{day}'  # Correctly format month and day
                day_data = self.plotting_df[self.plotting_df['Date'].dt.strftime('%Y-%m-%d') == day_str]
                netload_data[day] = day_data['TARG__NetLoad'].values
                global_min = min(global_min, netload_data[day].min())
                global_max = max(global_max, netload_data[day].max())

            ax = axs[idx]

            for i, day in enumerate(days):
                x = range(24)  # Assuming hourly data for each day
                ax.plot(x, netload_data[day], color=colors[i % len(colors)], label=f'Day {day}')

            # Add labels and legend
            ax.set_xlabel('Hour')
            ax.set_ylabel('Energy Load')
            ax.set_title(f'2015-{month}')
            ax.legend()
            ax.set_ylim(global_min, global_max)

    def plot_autocorrelation(self):
        """ Plots the Autocorrelation and Partial autocorrelation function of the target variable """

        # Plot the autocorrelation
        plot_acf(self.plotting_df['TARG__NetLoad'])
        plt.title("Autocorrelation")
        plt.xlabel('Hour')
        plt.ylabel('Load')
        plt.show()

        # Plot the partial autocorrelation
        plot_pacf(self.plotting_df['TARG__NetLoad'])
        plt.title("Partial autocorrelation")
        plt.xlabel('Hour')
        plt.ylabel('Load')
        plt.show()

    def plot_futu_correlation(self):
        """ Plots the correlation among the FUTU__ variables """

        # Retrieve the FUTU__ dataframe
        futures = self.plotting_df.filter(regex='^FUTU')

        # Compute the correlation matrix
        correlation_matrix = futures.corr()

        # Create a heatmap and plot
        plt.figure(figsize=(14, 12))  # Further increase figure size for better readability
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10},
                    cbar_kws={'shrink': 0.75})
        plt.title('Correlation Matrix', fontsize=18)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout(pad=1.0)
        plt.show()


class data_testing():

    '''Class used for testing normality of the target variable'''

    def __init__(self, ds: pd.core.frame.DataFrame, configs: dict, initial_date=datetime(year=2014, month=1, day=2),
                 last_date=datetime(year=2015, month=12, day=31)):
        '''Costructor of the class'''

        # Inputs:
        # ds:               dataset used for testing
        # configs:          configs
        # initial_date:     initial of the testing
        # last_date:        last date of the testing

        # Find the target variale in the time lag considered
        ds['Date'] = pd.to_datetime(ds['Date'])
        self.plotting_df = ds[(ds['Date'] >= initial_date) & (ds['Date'] <= last_date)]
        self.target = ['TARG__NetLoad' if configs['data_config'].task_name == 'NetLoad' else sys.exit(
            'Plotting data has not been defined for Price')]
        self.start_date = initial_date
        self.end_date = last_date

    def target_normality(self):
        '''Method that tests the normality assumption of the target variable'''

        # Perform shapiro test
        result=stats.shapiro(self.plotting_df['TARG__NetLoad'])
        print("-------------------------------------------------------------\n"
              "Shapiro test on target variable produced the following p-value: \n", result[1])


def remove_highly_correlated_features(df, threshold=0.9):
    ''' Removes from the dataset one feature from each pair of highly correlated FUTU features'''

    # Inputs:
    # df:           considered dataframe
    # threshold:    absolute value of the correlation threshold above which features are considered highly correlated

    # Outputs:
    # selected_df:  dataframe after having removed the highly correlated features

    # Retrieve the FUTU__ dataframe and store the other columns
    futures_columns = df.filter(regex='^FUTU').columns

    # Calculate the correlation matrix
    corr_matrix = df[futures_columns].corr().abs()

    # Save the column names to drop
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                colname = corr_matrix.columns[j]
                to_drop.add(colname)

    # Drop the identified columns and retrieve the columns to keep
    df_reduced = df.drop(columns=to_drop)
    cols_to_keep = df_reduced.columns

    # Return
    return cols_to_keep


class fourier_transform():

    '''Class which implements a fourier analysis in order to reduce the
        trend and seasonality effect of the target variable'''

    def __init__(self, period_min: int = 1,period_max: int = 10000,n_freq: int = 3):
        '''Constructor of the class'''

        self.period_min = period_min
        self.period_max = period_max
        self.n_freq = n_freq
        self.column_name = "TARG__NetLoad"
        self.predictors = ["trend", "seasonality"]

    def add_trend_term(self, pdf):
        '''Method that adds a trend variable to the dataframe'''

        # Inputs:
        # pdf:      dataframe considered

        # Outputs:
        # pdf:      dataframe with the new trend column

        # Compute the trend and add it to a new column
        pdf["trend"] = pdf.apply(lambda row: row.name +1, axis=1)

        # Return
        return pdf

    def add_fourier_seasonality_term(self, pdf, n_freq = None):
        '''Method that implements fourier analysis and stores the results in a dataframe'''

        # Inputs:
        # pdf:          dataframe considered
        # n_freq:       number of frequencies considered

        # Outputs:
        # fourier_output: dataframe containing the fourier terms
        # pdf:            original dataframe with relevant fourier terms

        # Assign the class variable to n_freq variable if None
        if n_freq is None:
            n_freq = self.n_freq

        # Add the trend
        pdf= self.add_trend_term(pdf)

        # Performs fourier transformation
        fft_output = fft.fft(pdf[self.column_name].to_numpy())
        amplitude = np.abs(fft_output)
        freq = fft.fftfreq(len(pdf[self.column_name].to_numpy()))

        # Filter out negative frequencies
        mask = freq >= 0
        freq = freq[mask]
        amplitude = amplitude[mask]

        # Determine peaks
        peaks = sig.find_peaks(amplitude[freq >= 0])[0]
        peak_freq = freq[peaks]
        peak_amplitude = amplitude[peaks]

        # Create dataframe containing necessary parameters
        fourier_output = pd.DataFrame()
        fourier_output["index"] = peaks
        fourier_output["freq"] = peak_freq
        fourier_output["amplitude"] = peak_amplitude
        fourier_output["period"] = 1 / peak_freq
        fourier_output["fft"] = fft_output[peaks]
        fourier_output["amplitude"] = fourier_output.fft.apply(lambda z: np.abs(z))
        fourier_output["phase"] = fourier_output.fft.apply(lambda z: np.angle(z))
        N = len(pdf.index)
        fourier_output["amplitude"] = fourier_output["amplitude"] / N
        fourier_output = fourier_output.sort_values("amplitude", ascending=False)
        fourier_output = fourier_output[fourier_output["period"] >= self.period_min]
        fourier_output = fourier_output[fourier_output["period"] <= self.period_max]

        # Store relevant parameters to the original dataframe
        pdf["freq"]=fourier_output["freq"]
        pdf["amplitude"]=fourier_output["amplitude"]

        # Turn our dataframe into a dictionary for easy lookup
        fourier_output_dict = fourier_output.to_dict("index")

        # Store the trend rows in a new independent variable
        pdf_temp = pdf[["trend"]]

        # Generate cosine waves for each amplitude, frequency and phase
        for key in islice(fourier_output_dict.keys(), n_freq):
            a = fourier_output_dict[key]["amplitude"]
            w = 2 * math.pi * fourier_output_dict[key]["freq"]
            p = fourier_output_dict[key]["phase"]
            pdf_temp[key] = pdf_temp["trend"].apply(
                            lambda t: a * math.cos(w * t + p))

        # Initialize a new column
        pdf_temp["FT_All"] = 0

        # Accumulate the contributions from the first n_freq Fourier transformed components
        for column in list(fourier_output.index)[:n_freq]:
            pdf_temp["FT_All"] = pdf_temp["FT_All"] + pdf_temp[column]

        # Convert and round the results and store them in a new column
        pdf["seasonality"] = pdf_temp["FT_All"].astype(float)
        pdf["seasonality"] = pdf["seasonality"].round(4)

        # Return
        return fourier_output, pdf


    def fit(self, training_dataset: pd.DataFrame):
        '''Method that fits a linear model the model to the training'''

        # Inputs:
        # training_dataset:     dataset used for training

        # Outputs:
        # self.fourier_training_dataset:   training residuals

        # Perform fourier analysis
        self.fourier_output, self.fourier_training_dataset = self.add_fourier_seasonality_term(training_dataset)

        # Extract the regressors and target variables
        X = self.fourier_training_dataset[self.predictors]
        y = self.fourier_training_dataset[self.column_name]

        # Define the model and fit it
        lm = LinearRegression()
        self.model = lm.fit(X, y)

        # Forecast baseline for the dataset
        self.fourier_training_dataset["baseline"] = self.model.predict(X)
        self.fourier_training_dataset["baseline"] = self.fourier_training_dataset["baseline"].round(4)

        # Compute the residuals
        self.training_residuals = self.fourier_training_dataset[self.column_name] - self.fourier_training_dataset["baseline"]

        # Return
        return self.fourier_training_dataset


    def predict(self, testing_dataset: pd.DataFrame):
        '''Method which performs testing'''

        # Inputs:
        # testing_dataset:     dataset used for testing

        # Outputs:
        # total_residuals:      training and testing residuals
        # total_baseline:       training and testing baseline

        # Perform fourier analysis on the testing
        fourier_output_testing, fourier_dataset_testing = self.add_fourier_seasonality_term(testing_dataset)

        # Extract the regressors
        X_testing = fourier_dataset_testing[self.predictors]

        # Perform the prediction using the fitted model
        y_hat = self.model.predict(X_testing)

        # Compute the total residuals and baseline
        total_residuals = pd.concat([self.training_residuals, fourier_dataset_testing[self.column_name] - y_hat])
        total_baseline = pd.concat([self.fourier_training_dataset["baseline"], pd.Series(y_hat, index=fourier_dataset_testing.index)])

        # Return
        return total_residuals, total_baseline


    def create_plots_fourier(self, fourier_dataset: pd.DataFrame):
        '''Method which plots the results of the Fourier analysis'''

        # Inputs:
        # fourier_dataset:      considered dataset

        # Restore the indexes
        fourier_dataset = fourier_dataset.reset_index()

        # Convert the date into appropriate format
        fourier_dataset["Date"] = pd.to_datetime(fourier_dataset["Date"], format="%Y-%m-%d")
        fourier_dataset = fourier_dataset.set_index("Date")

        # Plot
        fig, axs = plt.subplots(ncols=2, figsize=(30, 5))
        sns.lineplot(data=fourier_dataset, x="Date", y=self.column_name, ax=axs[0],
                     label=self.column_name, color="grey")
        sns.lineplot(x="Date", y="baseline", data=fourier_dataset, ax=axs[0],
                     label="baseline", color="black")
        fourier_dataset = fourier_dataset.sort_values('freq')
        axs[1].step(self.fourier_training_dataset['freq'], fourier_dataset['amplitude'], label='peaks amplitude', color='black', where='mid')
        axs[1].set_xlabel('Frequency')
        axs[1].set_ylabel('Amplitude')
        axs[1].legend()
        axs[0].legend()
        plt.show()

        # Plot seasonality
        self.plot_seasonality(fourier_dataset)

    def plot_seasonality(self, pdf):
        '''Method which plots the waves of the Fourier analysis'''

        # Inputs:
        # pdf:      considered dataset

        # Perform the fourier analysis
        pdf = pdf.reset_index()
        pdf = self.add_trend_term(pdf=pdf)
        fourier_output, pdf = self.add_fourier_seasonality_term(pdf, n_freq=len(pdf[self.column_name].to_numpy()))

        # Store th largest amplitudes
        top_amps = pdf.nlargest(self.n_freq, 'amplitude')
        top_freqs = top_amps['freq'].values
        top_amps = top_amps['amplitude'].values

        # Plot
        combined_y = np.zeros(10000)
        max_period = np.ceil(1 / np.min(top_freqs))
        t = np.linspace(0, max_period, 10000)
        for amp, freq in zip(top_amps, top_freqs):
            period = 1 / freq
            y = amp * np.sin(2 * np.pi * freq * t)
            combined_y = combined_y + y
            plt.plot(t, y, label=f'period: {period:.2f} h, Amp: {amp:.2f}')

        plt.xlabel('Time (h)')
        plt.ylabel('Amplitude')
        plt.title(f'Top {self.n_freq} Sine Waves')
        plt.legend()
        plt.grid(True)
        plt.show()

class HolidayAnalyzer:

    '''Class which implements the holiday modification method'''

    def __init__(self, df, year):
        '''Constructor of the class'''

        # Inputs:
        # df:               dataframe
        # year:             year of the analysis

        # Store the class variables
        self.orginal_df = df.copy()
        self.df = df.copy()
        self.year = year

        # Convert the date to a datetime object
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Retrieve the weekdays and filter dates between start_date and end_date
        self.df_weekdays = self.df[(self.df['CONST__DoW'] != 0) & (self.df['CONST__DoW'] != 6)]
        start_date = datetime(year, 1, 1)  # attention for the 2014 case
        end_date = datetime(year, 12, 31)
        self.df_weekdays = self.df_weekdays[(self.df_weekdays['Date'] >= start_date) & (self.df_weekdays['Date'] <= end_date)]

        # Store class variable
        self.high_load_dates = []

    def find_repeating_holidays(self):
        '''Method that finds the days over the year that are holidays'''

        # Retrieve unique holiday dates
        dates_list = self.df[self.df["CONST__Holiday"] == 1]["Date"].unique()

        # Extract day and month, and count occurrences
        date_counts = pd.Series([(date.month, date.day) for date in dates_list]).value_counts()

        # Find repeating holidays (day/month combinations that occur more than once)
        repeating_holidays = date_counts[date_counts > 1].index.tolist() # 2014, 2015, 2016 > 1 to cope with 1 Jan

        # Convert repeating holidays to datetime objects with the specific year
        self.repeating_holidays = [datetime(self.year, month, day) for month, day in repeating_holidays]


    def calculate_max_loads(self):
        '''Method that computes the maximum loads values for the holiday days in the considered year'''

        # Outputs:
        # daily_max_loads:      list containing the maximum load per holiday day

        daily_max_loads = []

        # Compute the maximum loads
        for date in self.repeating_holidays:
            df_day = self.df[self.df['Date'] == date]
            if not df_day.empty:
                max_value = df_day['TARG__NetLoad'].max()
                daily_max_loads.append((date, max_value))
        return daily_max_loads

    def calculate_monthly_statistics(self):
        '''Method that computes the days to be removed from the holiday set'''

        for month in range(1, 13):

            # Find the corresponding month
            self.df_weekdays['Date'] = pd.to_datetime(self.df_weekdays['Date'])
            month_data = self.df_weekdays[self.df_weekdays['Date'].dt.month == month]
            if not month_data.empty:

                # Compute the mean and standard deviation for the working days of the month and define the threshold
                mean_max = month_data.groupby(month_data['Date'].dt.date)['TARG__NetLoad'].max().mean()
                std_max = month_data.groupby(month_data['Date'].dt.date)['TARG__NetLoad'].max().std()
                threshold = mean_max - std_max

                # Find the days to be removed
                for date, max_value in self.daily_max_loads:
                    if date.month == month and max_value > threshold:
                        self.high_load_dates.append((date.month, date.day))
    def plot_max_loads_by_month(self):
        '''Method that plots the maximum of the load per day for each holiday'''

        # Build the dictionary
        max_values_by_month = {month: [] for month in range(1, 13)}
        for date, value in self.daily_max_loads:
            max_values_by_month[date.month].append((date, value))

        # Plot the maximum loads
        for month, date_max_pairs in max_values_by_month.items():
            if date_max_pairs:

                # Sort the date_max_pairs by the day of the month
                date_max_pairs.sort(key=lambda x: x[0].day)

                dates = [date for date, value in date_max_pairs]
                max_values = [value for date, value in date_max_pairs]
                plt.figure(figsize=(10, 6))
                plt.plot(dates, max_values, marker='o', linestyle='-',
                         label=f'{pd.to_datetime(month, format="%m").strftime("%B")}')
                plt.title(f'Maximum Net Load for {pd.to_datetime(month, format="%m").strftime("%B")} {self.year}')
                plt.xlabel('Date')
                plt.ylabel('Max Net Load')
                plt.legend()
                plt.show()
    def assign_holidays(self):
        '''Method that assigns the holidays to be labeled as working days'''

        for month, day in self.high_load_dates:
            self.df.loc[(self.df['Date'].dt.month == month) & (self.df['Date'].dt.day == day), 'CONST__Holiday'] = 0

    def analyze(self):
        '''Method that runs the steps of the class'''

        # Find the holiday days repeating during the years
        self.find_repeating_holidays()

        # Compute the max loads, plot them
        self.daily_max_loads = self.calculate_max_loads()
        self.calculate_monthly_statistics()
        self.plot_max_loads_by_month()

        # Assign the new holidays
        self.assign_holidays()

        # Assign the new column
        self.orginal_df['CONST__Holiday'] = self.df['CONST__Holiday']

        # Return
        return self.orginal_df


