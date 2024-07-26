# Introduction
Predicting time series is challenging, especially when dealing with non-stationary and volatile data, such as electricity load.  Our project aims to produce day-ahead probabilistic forecasts for electricity load. 
It was part of a competition for the Financial Engineering course final project, focused on developing the best possible probabilistic forecast with a particular emphasis on minimizing the Delta Coverage. Delta Coverage is an index that assesses the goodness of probabilistic forecasts by determining the fraction of observations that fall outside the predicted confidence interval (CI) for a given nominal level α. For instance, if 90% of the realized daily power consumptions fall within the 90% CI, then the CI is considered reliable. 
We also monitored other metrics, both pointwise (MSE, RMSE, sMAPE, MAE) and probabilistic (average Pinball Score and average Winkler Score) to ensure a robust forecast model.
The final result was evaluated on a separate test dataset.

# Our Achievement
We are proud to announce that our forecast was evaluated as the best among all competitors. Achieving this was particularly challenging due to several factors:

- High Volume of Research: We conducted extensive research, studying numerous papers to find the most suitable methods for our problem (no tutoring was available given the spirit of the competition)
- Multiple Trials and Models: We made numerous trials and tested various models to arrive at the best solution.
- Time Constraints: The project had a tight deadline, adding to the challenge.
- Strong Competition: We faced strong competition from other participants.

# Methodologies
For a better understanding of our work, please refer to the attached [View Report](Report.pdf). It contains:

- A deeper description of the problem
- Detailed methodologies
- Bibliography of referenced papers
- Instructions on how to use and run the code

# Further material
You can find a brief summary where we explain the main methodologies in this [video](https://www.youtube.com/watch?v=bAE_it7zSAg).



