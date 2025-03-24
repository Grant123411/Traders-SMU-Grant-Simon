# Week 3 Assignment - Traders@SMU Alpha Program

## Overview

This directory contains materials for Level 2, Week 3 of the Alpha Program.

## Learning Objectives

- Master advanced statistical methods for finance
- Understand stochastic processes in financial modeling
- Develop skills in econometric analysis

## Assignment Instructions

Complete the following tasks:

1. Task 1: [Description]
2. Task 2: [Description]
3. Task 3: [Description]

## Submission Guidelines

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime

# Fetch stock data for GLD and GDX (Gold ETF & Gold Miners ETF)
symbols = ["GLD", "GDX"]
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2011, 1, 1)

# Fetch data from Stooq (historical financial data)
prices = web.DataReader(symbols, "stooq", start, end)
prices = prices.sort_index()  # Stooq data comes in descending order, so we sort it
prices = prices["Close"]  # Keep only close prices

# Define X1 and X2
X1 = prices["GLD"]  # Independent variable (predictor)
X2 = prices["GDX"]  # Dependent variable (response)

# Plot both time series
plt.figure(figsize=(10, 5))
plt.plot(X1.index, X1.values, label='GLD')
plt.plot(X2.index, X2.values, label='GDX')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Perform Linear Regression to compute Beta
X1_const = sm.add_constant(X1)  # Add constant term for regression
model = sm.OLS(X2, X1_const).fit()  # Fit Ordinary Least Squares model

# Extract beta coefficient
beta = model.params["GLD"]
print(f"Beta (GLD -> GDX): {beta}")

# Compute residual spread Z = X2 - beta * X1
Z = X2 - beta * X1
Z.name = "Residual Spread"

# Plot residual spread
plt.figure(figsize=(10, 5))
plt.plot(Z.index, Z.values, label="Residual Spread (Z)")
plt.axhline(0, color='red', linestyle='dashed')  # Mean line
plt.xlabel("Time")
plt.ylabel("Residual")
plt.legend()
plt.show()

# Check for stationarity in residuals (Cointegration Test)
p_value = adfuller(Z)[1]
if p_value < 0.05:
    print(f"ADF p-value: {p_value} → Residuals are stationary (Cointegrated)")
else:
    print(f"ADF p-value: {p_value} → Residuals are NOT stationary (No cointegration)")

# Some possible alternate method I found using Jahansen cointegration test
coint_t, p_value, _ = coint(X1, X2)
print(f"Cointegration p-value: {p_value}")
