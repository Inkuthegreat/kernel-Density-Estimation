#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 10:39:21 2025

@author: sysadm
"""

import numpy as np
import matplotlib.pyplot as plt

# *****************************************
# Load samples
# *****************************************
samples = np.loadtxt("samples.csv", delimiter=",")
print(f"No. of samples loaded: {len(samples)}")

# *****************************************
# Histogram
# *****************************************
plt.figure(figsize=(10,6))
plt.hist(samples, bins=80, density=True, alpha=0.4, color="orange", edgecolor="gray", label="Histogram")

# **********************************************************************************
# Kernel Density Estimation - IMPLEMENT THIS FUNCTION 
# **********************************************************************************
def kde(x_points, data, bandwidth,ktype='gaussian'):

    None







# **********************************************************************************

# Bandwidth value
h = 0.1  
# points at which you will measure the density
x_grid = np.linspace(min(samples) - 1, max(samples) + 1, 1000)
ktype = 'gaussian'  # or 'uniform'
kde_values = kde(x_grid, samples, h, ktype)

# Plot KDE (UNCOMMENT THE LINE BELOW ONCE YOU HAVE IMPLEMENTED THE FUNCTION)
#### plt.plot(x_grid, kde_values, color="blue", linewidth=2, label=f"KDE (h={h})")

# *****************************************
# Formatting
# *****************************************
plt.title("Histogram and Kernel Density Estimation")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
