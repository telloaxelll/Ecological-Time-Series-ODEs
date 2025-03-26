"""" 
Author: AXEL MUNIZ TELLO, 
Date: 03/25/2025
Description: This script will be used to simulate the population dynamics of bobcats and rabbits using the Lotka-Volterra model
             for the applied mathematics challenge at UC Merced (2025)
"""
# Imported Libraries: 
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Save Plots in Directory: 
if not os.path.exists('plots'):
    os.makedirs('plots')
    
# Bobcat & Rabbit Datasets (Loaded):
bobcat_data = pd.read_csv('data/bobcatData.csv')
rabbit_data = pd.read_csv('data/rabbitData.csv')

# Clean Data:
rabbit_data.columns = ['Day', 'Rabbits']
bobcat_data.columns = ['Day', 'Bobcats']

# Extract Data:
t_rabbit = rabbit_data['Day'].values
x_values = rabbit_data['Rabbits'].values

t_bobcat = bobcat_data['Day'].values
y_values = bobcat_data['Bobcats'].values

##---INITIAL POPULATION PLOTS---## 
# Plot for Rabbit Population Over Time:
rabbit_data.plot(x='Day', y='Rabbits', kind='scatter') 
plt.title('Rabbit Population Over Time') 
plt.xlabel('Day')
plt.ylabel('Population')
plt.savefig('plots/rabbit_population_plot.png')
plt.show()

# Plot for Bobcat Population Over Time:
bobcat_data.plot(x='Day', y='Bobcats', kind='scatter')  
plt.title('Bobcat Population Over Time')
plt.xlabel('Day')
plt.ylabel('Population')
plt.savefig('plots/bobcat_population_plot.png') 
plt.show()
##-------------------------------##

##--INTERPOLATED DATA --## 
t_full = np.arrange(0, 90)
interp_x = interp1d(t_rabbit, x_values, kind='cubic', fill_value='extrapolate')
interp_y = interp1d(t_bobcat, y_values, kind='cubic', fill_value='extrapolate')

x_interp = interp_x(t_full)
y_interp = interp_y(t_full) 

##--ESTIMATED DERIVATIVES--##
dxdt = np.gradient(x_interp, t_full)
dydt = np.gradient(y_interp, t_full)

A1 = np.column_stack()