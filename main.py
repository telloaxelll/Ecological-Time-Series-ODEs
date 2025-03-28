"""
Author: AXEL MUNIZ TELLO
CONTRIBUTORS: LEANDRO OUVERNY, RYAN HSIAO
Date: 03/25/2025
Description: This script simulates the population dynamics of bobcats and rabbits using the Lotka-Volterra model
             for the Applied Mathematics Challenge at UC Merced (2025).
"""

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import os

# =========================
# FUNCTION DEFINITIONS
# =========================

# This function defines the Lotka-Volterra ODE system for predator-prey dynamics
def predator_prey(t, z, a, b, c, d):
    x, y = z
    dxdt = a * x - b * x * y
    dydt = d * x * y - c * y
    return [dxdt, dydt]

# This function computes the loss function for the optimization process
# It calcuates the mean squared error (MSE) between the observed and predicted populations
def loss(params):
    a, b, c, d = params
    try:
        sol = solve_ivp(
            lambda t, z: predator_prey(t, z, a, b, c, d),
            [t_common[0], t_common[-1]],
            [x_interp_norm[0], y_interp_norm[0]],
            t_eval=t_common,
            method='RK45'
        )

        if not sol.success or np.any(np.isnan(sol.y)) or sol.y.shape[1] != len(t_common):
            return np.inf

        x_model, y_model = sol.y
        mse_rabbits = np.mean((x_interp_norm - x_model) ** 2)
        mse_bobcats = np.mean((y_interp_norm - y_model) ** 2)
        return mse_rabbits + mse_bobcats

    except Exception:
        return np.inf

# =========================
# MAIN SCRIPT
# =========================

if not os.path.exists('plots'):
    os.makedirs('plots')

# Load population data
bobcat_data = pd.read_csv('data/bobcatData.csv')
rabbit_data = pd.read_csv('data/rabbitData.csv')

rabbit_data.columns = ['Day', 'Rabbits']
bobcat_data.columns = ['Day', 'Bobcats']

t_rabbit = rabbit_data['Day'].values
t_bobcat = bobcat_data['Day'].values
x_values = rabbit_data['Rabbits'].values
y_values = bobcat_data['Bobcats'].values

# Plot original data
rabbit_data.plot.scatter(x='Day', y='Rabbits', s=5)
plt.title('Rabbit Population Over Time')
plt.xlabel('Day')
plt.ylabel('Population')
plt.savefig('plots/rabbit_population_plot.png')
plt.close()

bobcat_data.plot.scatter(x='Day', y='Bobcats', s=5)
plt.title('Bobcat Population Over Time')
plt.xlabel('Day')
plt.ylabel('Population')
plt.savefig('plots/bobcat_population_plot.png')
plt.close()

# Interpolation onto common time grid
t_min = max(t_rabbit.min(), t_bobcat.min())
t_max = min(t_rabbit.max(), t_bobcat.max())
t_common = np.linspace(t_min, t_max, 200)

interp_rabbit = interp1d(t_rabbit, x_values, kind='linear', fill_value='extrapolate')
interp_bobcat = interp1d(t_bobcat, y_values, kind='linear', fill_value='extrapolate')

x_interp = interp_rabbit(t_common)
y_interp = interp_bobcat(t_common)

# Normalize interpolated data
x_interp_norm = x_interp / np.max(x_interp)
y_interp_norm = y_interp / np.max(y_interp)

# Plot normalized interpolated data
plt.plot(t_common, x_interp_norm, label='Normalized Rabbits')
plt.plot(t_common, y_interp_norm, label='Normalized Bobcats')
plt.title("Normalized Interpolated Populations")
plt.xlabel("Day")
plt.ylabel("Normalized Population")
plt.legend()
plt.grid(True)
plt.savefig("plots/interpolated_data.png")
plt.close()

# Fit model parameters
initial_guess = [0.2, 0.4, 0.5, 0.] # Parameter guesses for a, b, c, d cannot be 0, since a, b, c, d > 0 
bounds = [(0, 5), (0, 0.1), (0, 5), (0, 0.1)] # Ensures parameters are positive and not negative 
result = minimize(loss, initial_guess, bounds=bounds)

a_fit, b_fit, c_fit, d_fit = result.x
print(f"\nBest fit parameters:")
print(f"a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}, d = {d_fit:.4f}")
print(f"Final loss (MSE): {result.fun:.4f}")

# Simulate for 180 days (next 90 days forward)
t_total = np.linspace(0, 180, 181)
initial_conditions = [x_interp_norm[0], y_interp_norm[0]]
sol = solve_ivp(lambda t, z: predator_prey(t, z, a_fit, b_fit, c_fit, d_fit),
                [t_total[0], t_total[-1]],
                initial_conditions,
                t_eval=t_total,
                method='RK45')

x_pred, y_pred = sol.y

# Plot full prediction
plt.figure(figsize=(10, 6))
plt.plot(t_total, x_pred, label='Normalized Rabbits')
plt.plot(t_total, y_pred, label='Normalized Bobcats')
plt.axhline(200/np.max(y_interp), color='red', linestyle='--', label='Normalized Bobcat Threshold')
plt.xlabel("Day")
plt.ylabel("Normalized Population")
plt.title("Predicted Normalized Populations (180 Days)")
plt.legend()
plt.grid(True)
plt.savefig("plots/full_prediction.png")
plt.close()

# Check for bobcat overpopulation (in normalized scale)
if np.any(y_pred[90:] > 200 / np.max(y_interp)):
    print("Bobcat population exceeds 200 during the next 90 days!\n")
else:
    print("Bobcat population stays under control during the next 90 days.\n")