# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:53:03 2024

@author: Ana
"""

import numpy as np
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import pandas as pd


# DEFINITIONS
# y = (R(t)- r0)/r0
# x = t/tau

rho = 1000
r0 = 2.7*10**(-3)/2
d0 = 3.567*10**(-3)

fps = 5000
frame_no = 40
v0 = 3.6
gamma = 0.072
We = rho*d0*v0**2/gamma
eta = 10**-3


t =  [float(i/fps) for i in range(frame_no)]
tau = np.sqrt(rho*d0**3/6/gamma)

y = np.sqrt(2/3)*np.sqrt(We)*np.asarray(t)/tau*(1-np.asarray(t)/tau)**2
fitting  = np.sqrt(We)*np.asarray(t)*np.sin(np.pi/3)**2/tau*(1-np.cos(np.pi/3)*np.asarray(t)/tau)**2

folder = r'd:\Users\Ana\Desktop\Wonderful PhD\image_analisys\compute circularity\final functions to extract data from imapct videos'
df = pd.read_excel(folder +'\\radius_dynamics.xlsx')

# Replace 'your_column_name' with the actual name of the column you want to extract
column_values = df['R(t)-r0/r0'].tolist()

# Filter out non-numeric values if necessary
numeric_values = [value for value in column_values if isinstance(value, (int, float))]
time = [i/tau for i in t]

# plt.plot(time,y)
# plt.scatter(time, numeric_values)
# plt.ylabel('(R(t)-r0)/r0')
# plt.xlabel('t/tau')
# plt.legend(['Theory', 'Experiments'])
# Plotting
plt.figure(figsize=(8, 6))  # Create a new figure with specified size
plt.plot(time, numeric_values, 'b.', label='experiment')
plt.plot(time, fitting, 'r-', label='Fitted curve')
plt.plot(time,y, label = 'Theory')
plt.xlabel('t/tau')
plt.ylabel('Numeric Values')
plt.legend()
plt.show()



# SPLINE FIT
import numpy as np
from scipy.interpolate import splrep, BSpline

# Data
xdata = np.asarray(time)  # Assuming 'time' is defined elsewhere
ydata = numeric_values

tck = splrep(xdata, ydata, s=0)
tck_s = splrep(xdata, ydata, s=len(xdata))


plt.plot(xdata, BSpline(*tck)(xdata), '-', label='s=0')
plt.plot(xdata, BSpline(*tck_s)(xdata), '-', label=f's={len(xdata)}')
plt.plot(xdata, ydata, 'o')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))  # Create a new figure with specified size


from scipy.interpolate import UnivariateSpline

# Generate some example data
x = xdata
y = ydata

# Fit a spline to the data
spline = UnivariateSpline(x, y)

# Define the points where you want to evaluate the derivatives
x_new = xdata

# Compute the first and second derivatives
y_prime = spline.derivative(n=1)(x_new)
y_double_prime = spline.derivative(n=2)(x_new)

# Plot the original data
plt.scatter(x, y, color='red', label='Original Data')

# Plot the spline
plt.plot(x_new, spline(x_new), label='Spline')

# Plot the first derivative
plt.plot(x_new, y_prime, label='First Derivative')

# Plot the second derivative
plt.plot(x_new, y_double_prime, label='Second Derivative')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Spline and Its Derivatives')
plt.legend()
plt.grid(True)
plt.show()





# # Data
# xdata = np.asarray(time[0:int(len(time)/2)])  # Assuming 'time' is defined elsewhere
# ydata = numeric_values[0:int(len(time)/2)]

# # Initial guess
# x0 = np.array([1.0])

# # Standard deviation
# sigma = np.ones(len(xdata))

# # Function definition
# def func(x, a):
#     rho = 1000
#     d0 = 3.567 * 10**(-3)
#     v0 = 3.6
#     sigma_ = 0.072  # Renamed to avoid shadowing the outer 'sigma'
#     We = rho * d0 * v0**2 / sigma_

#     tau = np.sqrt(rho * d0**3 / 6 / sigma_)
#     return a *np.sqrt(2/3)* np.sqrt(We)*x/tau*(1-x/tau)**2
# # Fit parameters
# a, cov = optimization.curve_fit(func, xdata, ydata, x0, sigma)

# # Generate fitted curve
# fitting = func(xdata, a)

# # Plotting
# plt.figure(figsize=(8, 6))  # Create a new figure with specified size
# plt.plot(xdata, ydata, 'b.', label='Data')
# plt.plot(xdata, fitting, 'r-', label='Fitted curve')
# plt.plot(time,y, label = 'Theory')
# plt.xlabel('Time')
# plt.ylabel('Numeric Values')
# plt.legend()
# plt.show()

# # Print fitting parameters
# print("Fitting parameters:")
# print("a =", a)



# from scipy.optimize import minimize

# # Initial guess
# x0 = np.array([1, 1.0])

# # Define the fitting function
# def func(params, xdata, ydata):
#     a, b = params
#     rho = 1000
#     d0 = 3.567 * 10**(-3)
#     v0 = 3.6
#     sigma = 0.072
#     We = rho * d0 * v0**2 / sigma

#     tau = np.sqrt(rho * d0**3 / 6 / sigma)
#     residuals = ydata - (a * np.sqrt(We) * xdata / tau * (1 - xdata / tau)**2 + b)
#     return np.sum(residuals**2)

# # Perform the optimization
# result = minimize(func, x0, args=(xdata, ydata))

# # Extract the fitting parameters
# a_fit, b_fit = result.x

# # Generate fitted curve
# fitting = a_fit * np.sqrt(We) * xdata / tau * (1 - xdata / tau)**2 + b_fit

# # Plotting
# plt.figure(figsize=(8, 6))
# plt.plot(xdata, ydata, 'b.', label='Data')
# plt.plot(xdata, fitting, 'r-', label='Fitted curve')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Nonlinear Curve Fitting')
# plt.grid(True)
# plt.show()

# # Print fitting parameters
# print("Fitting parameters:")
# print("a =", a_fit)
# print("b =", b_fit)

# # Fitting the radius

# Radius = df['Radius[m]'].tolist()
# time = df['t[s]'].tolist()

# # Fit a polynomial curve (adjust the degree as needed)
# coefficients = np.polyfit(time, Radius, deg=4)
# poly = np.poly1d(coefficients)

# # Generate x values for the fitted curve
# x_fit = np.linspace(min(time), max(time), 100)
# y_fit = poly(x_fit)

# # Compute the derivative of the fitted polynomial
# derivative_poly = poly.deriv()

# # Generate y values for the derivative curve
# y_derivative = derivative_poly(x_fit)


# # Create two subplots
# fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# # Plot original data points and fitted curve
# axes[0].scatter(time, Radius, label='Data Points')
# axes[0].plot(x_fit, y_fit, color='red', label='Fitted Curve')
# axes[0].set_xlabel('time[s]')
# axes[0].set_ylabel('Radius[mm]')
# axes[0].set_title('Polynomial Curve Fitting')
# axes[0].legend()
# axes[0].grid(True)

# # Plot derivative curve
# axes[1].plot(x_fit, y_derivative, color='green')
# axes[1].set_xlabel('time[s]')
# axes[1].set_ylabel('velocity [mm/s]')

# axes[1].legend()
# axes[1].grid(True)

# We = [rho*r0*v**2/sigma for v in y_derivative]
# Re = [rho*r0*v/eta for v in y_derivative]
# # plot Weber
# axes[2].semilogy(x_fit, We, color='blue', label='We')
# axes[2].semilogy(x_fit, Re, color='orange', label='Re')
# axes[2].set_xlabel('time[s]')
# axes[2].set_ylabel('We')
# axes[2].legend()
# axes[2].grid(True)


