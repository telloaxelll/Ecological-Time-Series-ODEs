# Necessary Modules:
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt

# Save Plots in a Folder:
if not os.path.exists('plots'):
    os.makedirs('plots')
    
# Bobcat & Rabbit Datasets:
bobcat_data = pd.read_csv('data/bobcatData.csv')
rabbit_data = pd.read_csv('data/rabbitData.csv')

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