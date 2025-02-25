"""
PYTHON AND MACHINE LEARNING FOUNDATIONS
======================================
A step-by-step tutorial covering Python basics to ML implementation
"""

#############################
# PART 1: PYTHON BASICS
#############################

print("PART 1: PYTHON BASICS")
print("-" * 50)

# Variables and data types
integer_var = 10
float_var = 5.5
string_var = "Hello, Machine Learning"
boolean_var = True
list_var = [1, 2, 3, 4, 5]
dict_var = {"name": "Alice", "age": 30}

print(f"Integer: {integer_var}, Type: {type(integer_var)}")
print(f"Float: {float_var}, Type: {type(float_var)}")
print(f"String: {string_var}, Type: {type(string_var)}")
print(f"Boolean: {boolean_var}, Type: {type(boolean_var)}")
print(f"List: {list_var}, Type: {type(list_var)}")
print(f"Dictionary: {dict_var}, Type: {type(dict_var)}")
print()

# Control flow
print("Control Flow Examples:")
# If-else statement
x = 15
if x > 10:
    print("x is greater than 10")
else:
    print("x is less than or equal to 10")

# For loop
print("Loop through list:")
for item in list_var:
    print(f"  Item: {item}")

# List comprehension (powerful Python feature)
squares = [i**2 for i in range(1, 6)]
print(f"Squares using list comprehension: {squares}")

# Functions
def calculate_mean(numbers):
    """Calculate the mean of a list of numbers"""
    return sum(numbers) / len(numbers)

print(f"Mean of [1,2,3,4,5]: {calculate_mean(list_var)}")
print()

#############################
# PART 2: NUMPY BASICS
#############################

print("PART 2: NUMPY BASICS")
print("-" * 50)

import numpy as np

# Creating arrays
print("NumPy Arrays:")
array_1d = np.array([1, 2, 3, 4, 5])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
zeros_array = np.zeros((2, 3))
ones_array = np.ones((2, 2))
random_array = np.random.rand(2, 3)  # Random values between 0 and 1

print(f"1D Array: {array_1d}")
print(f"2D Array:\n{array_2d}")
print(f"Zeros Array:\n{zeros_array}")
print(f"Ones Array:\n{ones_array}")
print(f"Random Array:\n{random_array}")

# Array operations
print("\nArray Operations:")
print(f"Array + 5: {array_1d + 5}")
print(f"Array * 2: {array_1d * 2}")
print(f"Array squared: {array_1d ** 2}")
print(f"Mean: {np.mean(array_1d)}")
print(f"Sum: {np.sum(array_1d)}")
print(f"Standard Deviation: {np.std(array_1d)}")

# Shape manipulation
print("\nShape Manipulation:")
reshaped = array_1d.reshape(5, 1)
print(f"Reshaped array:\n{reshaped}")
print()

#############################
# PART 3: PANDAS BASICS
#############################

print("PART 3: PANDAS BASICS")
print("-" * 50)

import pandas as pd

# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 45],
    'City': ['New York', 'Boston', 'Chicago', 'Denver', 'Seattle'],
    'Salary': [60000, 70000, 80000, 90000, 100000]
}

df = pd.DataFrame(data)
print("DataFrame Basics:")
print(df)

# Basic DataFrame operations
print("\nDataFrame Operations:")
print("First 2 rows:")
print(df.head(2))
print("\nDataFrame Info:")
print(df.info())
print("\nDataFrame Statistics:")
print(df.describe())

# Data selection
print("\nData Selection:")
print("Selecting 'Age' column:")
print(df['Age'])
print("\nSelecting rows where Age > 30:")
print(df[df['Age'] > 30])

# Adding a new column
df['Experience'] = [3, 5, 8, 12, 15]
print("\nDataFrame with new column:")
print(df)
print()

#############################
# PART 4: MATPLOTLIB BASICS
#############################

print("PART 4: MATPLOTLIB BASICS")
print("-" * 50)

import matplotlib.pyplot as plt

print("Creating basic plots (figures will be displayed at the end)")
# Line plot
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, 'b-', label='sin(x)')
plt.title('Line Plot')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()

# Scatter plot
plt.subplot(2, 2, 2)
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y, color='red', alpha=0.5)
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')

# Bar plot
plt.subplot(2, 2, 3)
categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]
plt.bar(categories, values)
plt.title('Bar Plot')
plt.xlabel('Category')
plt.ylabel('Value')

# Histogram
plt.subplot(2, 2, 4)
data = np.random.randn(1000)
plt.hist(data, bins=30, alpha=0.7)
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
# We'll show all plots at the end to avoid interrupting the flow

#############################
# PART 5: MACHINE LEARNING BASICS
#############################

print("PART 5: MACHINE LEARNING BASICS")
print("-" * 50)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Key Machine Learning Concepts:")
print("1. Supervised Learning: Learning from labeled data")
print("   - Regression: Predicting continuous values")
print("   - Classification: Predicting categories")
print("2. Unsupervised Learning: Finding patterns in unlabeled data")
print("   - Clustering: Grouping similar data points")
print("   - Dimensionality Reduction: Simplifying data while preserving information")
print("3. ML Workflow: Data preparation → Model training → Evaluation → Prediction")

# Generate synthetic data for linear regression
print("\nGenerating synthetic data for our ML example:")
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # Feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Target = 4 + 3X + noise

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Create a DataFrame for better visualization
data_df = pd.DataFrame(np.hstack((X, y)), columns=['Feature', 'Target'])
print("\nFirst 5 rows of our data:")
print(data_df.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

#############################
# PART 6: LINEAR REGRESSION
#############################

print("\nPART 6: LINEAR REGRESSION IMPLEMENTATION")
print("-" * 50)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Model parameters
print("Model Parameters:")
print(f"Slope (Coefficient): {model.coef_[0][0]:.3f}")  # Should be close to 3
print(f"Intercept: {model.intercept_[0]:.3f}")  # Should be close to 4

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# Create final visualization
plt.figure(figsize=(10, 6))

# Plot the original data
plt.scatter(X_test, y_test, color='blue', alpha=0.7, label='Actual Data')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')

# Add a title and labels
plt.title('Linear Regression Model')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Show all plots
plt.show()

print("\nLesson complete! You've learned Python basics and implemented your first ML model.")
# Import necessary libraries
import numpy as np              # For numerical operations and array manipulation
import pandas as pd            # For data manipulation and analysis
import matplotlib.pyplot as plt # For data visualization
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.linear_model import LinearRegression     # The linear regression model
from sklearn.metrics import mean_squared_error        # For model evaluation

# Data Generation
# --------------
# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic feature data (x)
# Creates 100 random numbers between 0 and 2 (multiplied by 2)
# reshape into column vector with shape (100,1)
x = 2 * np.random.rand(100, 1)

# Generate synthetic target data (y)
# Using the equation: y = 4 + 3x + random_noise
# - 4 is the intercept
# - 3 is the slope
# - np.random.randn(100,1) adds random noise from normal distribution
y = 4 + 3 * x + np.random.randn(100, 1)

# Data Preparation
# --------------
# Combine x and y into a pandas DataFrame for better data handling
# np.hstack concatenates arrays horizontally
data = pd.DataFrame(np.hstack((x, y)), columns=['Feature', 'Target'])

# Split the dataset into training and testing sets
# - test_size=0.2 means 20% data for testing, 80% for training
# - random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Training
# -------------
# Initialize the linear regression model
model = LinearRegression()

# Fit the model using training data
# This step calculates the optimal slope and intercept
model.fit(X_train, y_train)

# After model training, add:
print(f"Model Coefficients (Slope): {model.coef_[0][0]:.3f}")
print(f"Model Intercept: {model.intercept_[0]:.3f}")
print(f"R² Score: {model.score(X_test, y_test):.3f}")

# Model Prediction and Evaluation
# -----------------------------
# Use the trained model to make predictions on test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) to evaluate model performance
# MSE = average of squared differences between predicted and actual values
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualization
# ------------
# Create a scatter plot of actual test data
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# Plot the regression line using predicted values
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')

# Add labels and title to make the plot more informative
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.title('Simple Linear Regression')

# Display the plot
plt.show()

