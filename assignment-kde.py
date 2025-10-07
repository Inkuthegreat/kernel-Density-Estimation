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



def gauss(x):
    return (np.exp(-(x**2)/2)/np.sqrt(2*np.pi))

def uform(x):
    return (0.5*(np.abs(x)<=1))

# **********************************************************************************
# Kernel Density Estimation - IMPLEMENT THIS FUNCTION 
# **********************************************************************************
def kde(x_points, data, bandwidth,ktype='gaussian'):
    n=len(data)
    density_estimate=np.zeros_like(x_points,dtype=float)
    for i in range(len(x_points)):
        x=x_points[i]
        u=(x-data)/bandwidth
        if ktype == 'gaussian':
            k_value=gauss(u)
        elif ktype == 'uniform':
            k_value=uform(u)
        density_estimate[i]=np.sum(k_value)/(n*bandwidth)
    return density_estimate








# **********************************************************************************

# Bandwidth value
h = 0.1  
# points at which you will measure the density
x_grid = np.linspace(min(samples) - 1, max(samples) + 1, 1000)
ktype = 'gaussian'  # or 'uniform'
kde_values_g = kde(x_grid, samples, h, ktype)

ktype = 'uniform'
kde_values_u = kde(x_grid, samples, h, ktype)





# Plot KDE (UNCOMMENT THE LINE BELOW ONCE YOU HAVE IMPLEMENTED THE FUNCTION)
plt.plot(x_grid, kde_values_g, color="blue", linewidth=2, label=f"KDE gaussian (h={h})")

plt.plot(x_grid, kde_values_u, color="green", linewidth=2, label=f"KDE uniform (h={h})")


h = 0.3

ktype = 'gaussian'  # or 'uniform'
kde_values_g_new = kde(x_grid, samples, h, ktype)

ktype = 'uniform'
kde_values_u_new = kde(x_grid, samples, h, ktype)


plt.plot(x_grid, kde_values_g_new, color="red", linewidth=2, label=f"KDE gaussian (h={h})")

plt.plot(x_grid, kde_values_u_new, color="black", linewidth=2, label=f"KDE uniform (h={h})")
# *****************************************
# Formatting
# *****************************************
plt.title("Histogram and Kernel Density Estimation")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
