import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('top_six_economies.csv')
df = df[df['Country Name'] == 'Japan']

# Part 1
# Extract the data
x = df['Year'].values
y_true = df['GDP (current US$)'].values

# Exponential model function
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Find the best-fitted parameters by maximizing R-squared
best_r_squared = -np.inf
best_params = None

# Range of values for the parameters
a_range = np.linspace(1e11, 1e13, 100)
b_range = np.linspace(0.01, 0.1, 100)

# Iterate over parameter combinations
for a_guess in a_range:
    for b_guess in b_range:
        try:
            # Fit the exponential model to the data
            params, covariance = curve_fit(exponential_model, x, y_true, p0=[a_guess, b_guess], maxfev=10000)
            
            # Extract the parameters
            a_fit, b_fit = params

            # Predicted values using the fitted model
            y_pred = exponential_model(x, a_fit, b_fit)

            # Calculate R-squared value
            residuals = y_true - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Update best parameters if R-squared is improved
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_params = (a_fit, b_fit)
        except RuntimeError:
            # Catch RuntimeError and continue with the next iteration
            continue

# Unpack the best-fitted parameters
a_fit, b_fit = best_params

# Predicted values using the best-fitted model
y_pred = exponential_model(x, a_fit, b_fit)

# Plot the original data and the best-fitted exponential curve for GDP
plt.figure()
plt.scatter(x, y_true, label=f'Original Data\nExponential Fit: a={a_fit:.2e}, b={b_fit:.2e}\nR-squared: {best_r_squared:.4f}')
plt.plot(x, y_pred, color='red', linestyle='-', linewidth=2, label='Indicator Line')
plt.xlabel('Year')
plt.ylabel('GDP (current US$)')
plt.legend()
plt.title('Best-Fitted Exponential to GDP Data with Year as X-axis')

# Part 2
# Extract the data
x_unemployment = df['Unemployment, total (% of total labor force) (modeled ILO estimate)'].values
y_gdp = df['GDP (current US$)'].values

# Find the best-fitted parameters for GDP
best_r_squared_gdp = -np.inf
best_params_gdp = None

# Range of values for the parameters
a_range = np.linspace(1e11, 1e14, 100)
b_range = np.linspace(-0.1, 0.1, 100)

# Iterate over parameter combinations for GDP
for a_guess in a_range:
    for b_guess in b_range:
        try:
            # Fit the exponential model to the GDP data
            params, covariance = curve_fit(exponential_model, x_unemployment, y_gdp, p0=[a_guess, b_guess], maxfev=10000)
            
            # Extract the parameters
            a_fit, b_fit = params

            # Predicted values using the fitted model
            y_pred_gdp = exponential_model(x_unemployment, a_fit, b_fit)

            # Calculate R-squared value for GDP
            residuals = y_gdp - y_pred_gdp
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_gdp - np.mean(y_gdp))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Update best parameters if R-squared is improved for GDP
            if r_squared > best_r_squared_gdp:
                best_r_squared_gdp = r_squared
                best_params_gdp = (a_fit, b_fit)
        except RuntimeError:
            # Catch RuntimeError and continue with the next iteration
            continue

# Unpack the best-fitted parameters for GDP
a_fit_gdp, b_fit_gdp = best_params_gdp

# Predicted values using the best-fitted model for GDP
y_pred_gdp = exponential_model(x_unemployment, a_fit_gdp, b_fit_gdp)

# Plot the original data and the best-fitted exponential curve for GDP
plt.figure()
plt.scatter(x_unemployment, y_gdp, label=f'Original GDP Data\nExponential Fit: a={a_fit_gdp:.2e}, b={b_fit_gdp:.2e}\nR-squared: {best_r_squared_gdp:.4f}')
plt.plot(x_unemployment, y_pred_gdp, color='red', linestyle='-', linewidth=2, label='Indicator Line')
plt.xlabel('Unemployment (% of total labor force)')
plt.ylabel('GDP (current US$)')
plt.legend()
plt.title('Best-Fitted Exponential to GDP Data with Unemployment as X-axis')

# Make predictions for specific years
years_to_predict = [2025, 2030, 2035]  # Change these to the years you want to predict
for year in years_to_predict:
    predicted_gdp = exponential_model(year, a_fit, b_fit)
    print(f"Predicted GDP for the year {year}: {predicted_gdp:.2f} (in current US$)")

# Make predictions for specific unemployment rates
unemployment_rates_to_predict = [3, 5, 7]  # Change these to the unemployment rates you want to predict
for rate in unemployment_rates_to_predict:
    predicted_gdp_unemployment = exponential_model(rate, a_fit_gdp, b_fit_gdp)
    print(f"Predicted GDP for an unemployment rate of {rate}%: {predicted_gdp_unemployment:.2f} (in current US$)")


plt.show()
print("Exponential Fit Done!!!")