# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:30:46 2024

@author: Aswini
"""

import numpy as np
# Import NumPy, a powerful library for numerical operations
import matplotlib.pyplot as plt
# Import the matplotlib.pyplot as plt, used to create plots and charts
from scipy.stats import norm
# Import the integrate module from SciPy for numerical integration
import pandas as pd
# Import the pandas library as pd, used for analysis

# Read data from the CSV file into a Pandas DataFrame
data = pd.read_csv('data3.csv', header=None, names=['Salaries'])
df = data['Salaries']

# Create a larger figure
plt.figure(figsize=(8, 6))

# Plot a histogram of salary data
plt.hist(df, bins=30, density=True, alpha=0.6, color='g', label='Histogram')

# Fit a normal distribution to the data get the mean
mu, _ = norm.fit(df)  

# Generate values for x-axis within the observed range of salaries
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

# Calculate the probability density function for the fitted normal distribution
p = norm.pdf(x, mu)

# Plot of the fitted normal distribution
plt.plot(x, p, 'k', linewidth=2)

# Calculate the mean annual salary (~W)
mean_salary = np.mean(df)

# Calculate X such that 33% of people have a salary below X
x_percentile = np.percentile(df, 33)

# Plot vertical lines to represent mean_salary and x_percentile on the graph
plt.axvline(mean_salary, color='blue', linestyle='dashed', linewidth=2, 
            label='Mean ($\~{W}$): %.2f' % mean_salary)
plt.axvline(x_percentile, color='red', linestyle='dashed', linewidth=2, 
            label='X (33% below): {:.2f}'.format(x_percentile))

# Display legend and show the plot
plt.legend()
plt.title('Annual Salary Distribution')
plt.xlabel('Annual Salary')
plt.ylabel('Probability Density')

#Display the plot
plt.show()