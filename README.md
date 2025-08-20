# MonteCasino Probabilistic Modeling Library

A high-performance Python library for probabilistic modeling and Monte Carlo simulation using a domain-specific language (DSL) with natural mathematical syntax.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Basic Usage](#basic-usage)
- [Random Variables](#random-variables)
- [Arithmetic Operations](#arithmetic-operations)
- [Advanced Features](#advanced-features)
- [Visualization](#visualization)
- [Performance](#performance)
- [Examples](#examples)
- [API Reference](#api-reference)

## Overview

MonteCasino enables you to build complex probabilistic models using familiar mathematical operators. The library compiles expressions into efficient bytecode executed by a custom virtual machine, providing fast Monte Carlo simulations with memory-efficient statistical summaries.

### Key Features

- **Natural Syntax**: Use standard Python operators (+, -, *, /, etc.) with random variables
- **High Performance**: Cython-optimized virtual machine for fast execution
- **Memory Efficient**: T-digest algorithm for bounded memory statistical summaries
- **Composable**: Build complex models from simple components
- **Extensible**: Easy to add new distributions and operations

## Installation

```bash
pip install montecasino
```

For development installation:
```bash
git clone <repository-url>
cd casino
pip install -e .
```

## Quick Start

```python
import montecasino as mc
import numpy as np

# Define random variables
x = mc.Normal(0, 1)        # Standard normal distribution
y = mc.Normal(5, 2)        # Normal with mean=5, std=2

# Combine with arithmetic operations
total = x + y               # Sum of two normals
product = x * y            # Product of distributions
power = x ** 2             # Squared distribution

# Run Monte Carlo simulation
result = total.compute(samples=10000)

# Access statistics
print(f"Median: {result.quantile(0.5)}")
print(f"95th percentile: {result.quantile(0.95)}")
print(f"Range: [{result.lower()}, {result.upper()}]")

# Visualize results
mc.plot(result)
```

## Core Concepts

### Random Variables

Random variables are the building blocks of probabilistic models. They represent uncertain quantities that can be combined using mathematical operations.

### Expression Trees

Operations between random variables create expression trees that capture the computational structure of your model.

### Virtual Machine

The library compiles expression trees into bytecode executed by a stack-based virtual machine optimized for Monte Carlo simulation.

### T-Digest

Statistical results are stored using the t-digest algorithm, which provides accurate quantile estimates with bounded memory usage.

## Basic Usage

### Creating Random Variables

```python
import montecasino as mc

# Normal distribution
normal = mc.Normal(mean=0, stdev=1)

# Random integers
dice = mc.RandInt(1, 6)  # Requires bounds as separate variables
low = mc.Constant(1)
high = mc.Constant(6)
dice = mc.RandInt(low, high)

# From empirical data
import numpy as np
data = np.random.normal(100, 15, 1000)
empirical = mc.fromArray(data)

# From quantiles
quantiles = mc.Quantiles(10, 20, 30, 40, 50)  # 0%, 25%, 50%, 75%, 100% quantiles
```

### Running Simulations

```python
# Single sample
single_value = normal.sample()

# Monte Carlo simulation (returns DigestVariable)
result = normal.compute(samples=10000, maxBins=32)

# Access statistics
median = result.quantile(0.5)
mean_approx = result.quantile(0.5)  # For symmetric distributions
p95 = result.quantile(0.95)
```

## Random Variables

### Normal Distribution

```python
# Standard normal (mean=0, std=1)
standard = mc.Normal()

# Custom parameters
custom = mc.Normal(mean=100, stdev=15)
```

### Random Integers

```python
# Uniform random integers
low = mc.Constant(1)
high = mc.Constant(10)
uniform_int = mc.RandInt(low, high)
```

### Empirical Distributions

```python
# From historical data
historical_data = [1.2, 1.5, 0.8, 2.1, 1.9, 1.1]
historical_dist = mc.fromArray(historical_data)

# From quantile specifications
expert_opinion = mc.Quantiles(0, 10, 25, 40, 50)  # Expert-defined quantiles
```

## Arithmetic Operations

### Basic Operations

```python
x = mc.Normal(0, 1)
y = mc.Normal(5, 2)

# Arithmetic
sum_xy = x + y           # Addition
diff_xy = x - y          # Subtraction
product_xy = x * y       # Multiplication
ratio_xy = x / y         # Division
power_x = x ** 2         # Exponentiation
remainder = x % y        # Modulo
floor_div = x // y       # Floor division
```

### Comparisons

```python
# Logical operations (return 1.0 or 0.0)
indicator1 = x < 0       # Less than
indicator2 = x <= 0      # Less than or equal
indicator3 = x > 0       # Greater than  
indicator4 = x >= 0      # Greater than or equal

# Example: Probability that sum exceeds threshold
prob_exceed = (x + y) > 10
result = prob_exceed.compute()
probability = result.quantile(0.5)  # Average of 0s and 1s
```

### Constants

```python
# Use constants in operations
x = mc.Normal(0, 1)
scaled = x * 10 + 5      # Scale and shift

# Or explicitly create constants
multiplier = mc.Constant(10)
offset = mc.Constant(5)
scaled_explicit = x * multiplier + offset
```

## Advanced Features

### Summations

Model scenarios requiring multiple random events:

```python
# Sum of dice rolls
die = mc.RandInt(mc.Constant(1), mc.Constant(6))
sum_of_10_dice = mc.Summation(nTerms=10, term=die)

# Portfolio modeling
stock_return = mc.Normal(0.08, 0.2)
num_stocks = mc.Constant(20)
portfolio = mc.Summation(nTerms=num_stocks, term=stock_return)
```

### Array Operations

Sum subsets of arrays:

```python
# Create portfolio of 20 assets
returns = [mc.Normal(0.08, 0.15) for _ in range(20)]

# Sum only the first 10 assets
partial_portfolio = mc.ArraySum(
    array=returns,
    start=0,
    end=10
)
```

### Min/Max Operations

```python
# Maximum of multiple variables
x1 = mc.Normal(0, 1)
x2 = mc.Normal(1, 1)
x3 = mc.Normal(-1, 1)

maximum = mc.max(x1, x2, x3)
minimum = mc.min(x1, x2, x3)
```

### Conditional Logic

```python
# Value at Risk (VaR) calculation
portfolio_return = mc.Normal(0.05, 0.15)
loss = -portfolio_return

# Indicator for losses exceeding 10%
extreme_loss = loss > 0.10
result = extreme_loss.compute()

# Probability of extreme loss
var_prob = result.quantile(0.5)
```

## Visualization

### Basic Plotting

```python
import montecasino as mc

# Create and simulate a model
x = mc.Normal(0, 1)
y = mc.Normal(2, 1)
combined = x + y

result = combined.compute(samples=10000)

# Plot histogram
mc.plot(result, nBins=30)
```

### Custom Plotting

```python
import matplotlib.pyplot as plt

# Custom styling
fig, ax = plt.subplots(figsize=(10, 6))
mc.plot(result, nBins=25, ax=ax, color='skyblue', alpha=0.7)
ax.set_title('Combined Distribution')
ax.set_xlabel('Value')
ax.set_ylabel('Probability')
plt.show()
```

## Performance

### Optimization Tips

1. **Batch Simulations**: Use higher `samples` parameter for better accuracy
2. **Memory Management**: Adjust `maxBins` parameter based on accuracy needs
3. **Reuse Results**: Store computed DigestVariables for reuse in multiple models

```python
# Efficient approach
base_distribution = mc.Normal(0, 1).compute(samples=100000, maxBins=64)

# Reuse in multiple models
model1 = base_distribution + 10
model2 = base_distribution * 2
model3 = base_distribution ** 2

results = [model.compute() for model in [model1, model2, model3]]
```

### Memory Usage

The t-digest algorithm ensures bounded memory usage regardless of sample size:

```python
# These use similar memory
small_simulation = mc.Normal(0, 1).compute(samples=1000)
large_simulation = mc.Normal(0, 1).compute(samples=1000000)
```

## Examples

### Portfolio Risk Assessment

```python
import montecasino as mc
import numpy as np

# Define asset returns
tech_stock = mc.Normal(0.12, 0.25)      # High return, high volatility
bond = mc.Normal(0.04, 0.05)            # Low return, low volatility
real_estate = mc.Normal(0.08, 0.15)     # Medium return, medium volatility

# Portfolio weights
portfolio = (tech_stock * 0.5 + 
             bond * 0.3 + 
             real_estate * 0.2)

# Simulate portfolio returns
result = portfolio.compute(samples=50000)

# Risk metrics
print(f"Expected return (median): {result.quantile(0.5):.3f}")
print(f"95% VaR: {-result.quantile(0.05):.3f}")
print(f"99% VaR: {-result.quantile(0.01):.3f}")

# Visualize
mc.plot(result, nBins=50)
```

### Insurance Claim Modeling

```python
# Claim frequency (number of claims)
num_claims = mc.RandInt(mc.Constant(0), mc.Constant(100))

# Claim severity (amount per claim)
claim_amount = mc.Normal(5000, 2000)

# Total claims using summation
total_claims = mc.Summation(nTerms=num_claims, term=claim_amount)

# Add administrative costs
admin_cost = mc.Normal(1000, 200)
total_cost = total_claims + admin_cost

# Simulate
result = total_cost.compute(samples=25000)

print(f"Expected total cost: ${result.quantile(0.5):,.0f}")
print(f"95th percentile: ${result.quantile(0.95):,.0f}")
```

### Option Pricing (Simplified Black-Scholes)

```python
# Parameters
S0 = 100          # Current stock price
K = 105           # Strike price
T = 1.0           # Time to expiration
r = 0.05          # Risk-free rate
sigma = 0.2       # Volatility

# Stock price at expiration (geometric Brownian motion approximation)
Z = mc.Normal(0, 1)
ST = S0 * mc.exp((r - 0.5 * sigma**2) * T + sigma * (T**0.5) * Z)

# Call option payoff
call_payoff = mc.max(ST - K, 0)

# Present value
call_price = call_payoff * mc.exp(-r * T)

# Note: This is a simplified example. 
# Real option pricing would use more sophisticated methods.
```

### Quality Control

```python
# Manufacturing process
measurement_error = mc.Normal(0, 0.1)
true_dimension = mc.Constant(10.0)
measured_dimension = true_dimension + measurement_error

# Acceptance criteria
in_spec = (measured_dimension >= 9.9) & (measured_dimension <= 10.1)

# Yield calculation
result = in_spec.compute(samples=10000)
yield_rate = result.quantile(0.5)

print(f"Expected yield: {yield_rate:.1%}")
```

## API Reference

### Core Classes

#### RandomVariable
Base class for all random variables and expressions.

**Methods:**
- `sample()`: Generate single random sample
- `compute(samples=10000, maxBins=32)`: Run Monte Carlo simulation
- `printTree()`: Display expression tree structure

#### Normal(mean=0, stdev=1)
Normal distribution random variable.

#### RandInt(low, high)
Uniform random integer generator.

#### Summation(nTerms, term)
Mathematical summation operation.

#### Quantiles(*args)
Distribution defined by quantile values.

#### ArraySum(array, start, end)
Sum subset of array elements.

### Utility Functions

#### fromArray(array, maxBins=32)
Create DigestVariable from empirical data.

#### plot(digest, nBins=20, lower=None, upper=None, width=0.8, ax=None)
Plot histogram of distribution.

#### max(*args), min(*args)
Elementwise maximum/minimum of multiple variables.

### DigestVariable Methods

#### quantile(q)
Compute quantile for probability q (0 ≤ q ≤ 1).

#### cdf(k)
Compute cumulative distribution function at value k.

#### lower(), upper()
Get minimum and maximum values in distribution.

---

## Contributing

Contributions are welcome! Please see the contributing guidelines for details on how to submit improvements, bug reports, and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.