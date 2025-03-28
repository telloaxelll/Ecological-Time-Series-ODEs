"""" 
Author: AXEL MUNIZ TELLO, LEANDRO OUVERNY, RYAN HSIAO
Date: 03/25/2025
Description: This script will be used to simulate the population dynamics of bobcats and rabbits using the Lotka-Volterra model
             for the applied mathematics challenge at UC Merced (2025)
Note: Please refer to the README.md file for more information on the project and the workings of the code.
"""

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import os 

if not os.path.exists('plots'):
    os.makedirs('plots')
    

def predator_prey (t, z, a, b, c, d):
    x,y = z
    dxdt = a*x - b*x*y
    dydt = d*x*y - c*y
    return [dxdt, dydt]
    
# Loaded Data for Bobcat and Rabbit Populations:
bobcat_data = pd.read_csv('data/bobcatData.csv')
rabbit_data = pd.read_csv('data/rabbitData.csv')

# Organizing Data:
rabbit_data.columns = ['Day', 'Rabbits']
bobcat_data.columns = ['Day', 'Bobcats']

# Extract Data from Columns:
t_rabbit = rabbit_data['Day'].values
t_bobcat = bobcat_data['Day'].values
x_values = rabbit_data['Rabbits'].values
y_values = bobcat_data['Bobcats'].values

# Plot for Rabbit Population Over Time:
rabbit_data.plot.scatter(x='Day', y='Rabbits', s=5) 
plt.title('Rabbit Population Over Time') 
plt.xlabel('Day')
plt.ylabel('Population')
plt.savefig('plots/rabbit_population_plot.png')
plt.close()

# Plot for Bobcat Population Over Time:
bobcat_data.plot.scatter(x='Day', y='Bobcats', s=5)  
plt.title('Bobcat Population Over Time')
plt.xlabel('Day')
plt.ylabel('Population')
plt.savefig('plots/bobcat_population_plot.png') 
plt.close()

######-BEGINNING OF METHOD-######
"""
In order to be able to solve these ODEs we we are going to need more data points. However, we 
are going to need to interpolate the data points we currently have in order to get more data points
that we can work with. 
"""
# Interpolation:
# Create a common time grid (ex) 90 days evenly spaced)
t_min = max(t_rabbit.min(), t_bobcat.min())
t_max = min(t_rabbit.max(), t_bobcat.max())
t_common = np.linspace(t_min, t_max, 200)

# Interpolate both datasets onto common time grid
interp_rabbit = interp1d(t_rabbit, x_values, kind='linear', fill_value='extrapolate')
interp_bobcat = interp1d(t_bobcat, y_values, kind='linear', fill_value='extrapolate')

x_interp = interp_rabbit(t_common) # x_interp = rabbit population => x(t)
y_interp = interp_bobcat(t_common) # y_interp = bobcat population => y(t)

# Plot for Interpolated Rabbit Population:
plt.plot(t_common, x_interp, label='Interpolated Rabbits')
plt.plot(t_common, y_interp, label='Interpolated Bobcats')
plt.title("Interpolated Populations")
plt.xlabel("Day")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.savefig("plots/interpolated_data.png")
plt.close()