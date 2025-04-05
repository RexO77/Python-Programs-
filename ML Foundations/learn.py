"""
MACHINE LEARNING FOUNDATIONS: A STRUCTURED LEARNING PATH
========================================================
This tutorial covers essential Python skills and libraries needed for machine learning,
progressing from basic concepts to practical ML implementation.
"""

#############################
# SECTION 1: PYTHON ESSENTIALS FOR ML
#############################

print("SECTION 1: PYTHON ESSENTIALS FOR ML")
print("=" * 60)
print("These Python fundamentals are particularly important for ML work")
print("-" * 60)

# Variables and data types commonly used in ML
integer_var = 10
float_var = 5.5
string_var = "Hello, Machine Learning"
boolean_var = True
list_var = [1, 2, 3, 4, 5]
dict_var = {"name": "Alice", "age": 30}

# Displaying variable types - understanding data types is critical in ML
print(f"Integer: {integer_var}, Type: {type(integer_var)}")
print(f"Float: {float_var}, Type: {type(float_var)}")
print(f"String: {string_var}, Type: {type(string_var)}")
print(f"Boolean: {boolean_var}, Type: {type(boolean_var)}")
print(f"List: {list_var}, Type: {type(list_var)}")
print(f"Dictionary: {dict_var}, Type: {type(dict_var)}")
print()

# Control flow - essential for data processing and algorithm implementation
print("Control Flow Examples:")
# If-else statement - used in model selection and evaluation
x = 15
if x > 10:
    print("x is greater than 10")
else:
    print("x is less than or equal to 10")

# For loop - commonly used for iterating through datasets
print("Loop through list:")
for item in list_var:
    print(f"  Item: {item}")

# List comprehension - powerful for data transformation
squares = [i**2 for i in range(1, 6)]
print(f"Squares using list comprehension: {squares}")

# Functions - fundamental for code organization in ML projects
def calculate_mean(numbers):
    """Calculate the mean of a list of numbers"""
    return sum(numbers) / len(numbers)

print(f"Mean of [1,2,3,4,5]: {calculate_mean(list_var)}")
print()

#############################
# SECTION 2: DATA MANIPULATION TOOLS
#############################

print("SECTION 2: DATA MANIPULATION TOOLS")
print("=" * 60)
print("NumPy and Pandas are the foundation for data handling in ML")
print("-" * 60)

# PART A: NUMPY - THE NUMERICAL FOUNDATION OF ML
print("\nPART A: NUMPY - THE NUMERICAL FOUNDATION OF ML")
print("-" * 50)

import numpy as np

# Creating arrays - ML data is typically stored in arrays
print("NumPy Arrays:")
array_1d = np.array([1, 2, 3, 4, 5])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])  # 2D arrays represent tables or matrices
zeros_array = np.zeros((2, 3))  # Used for initializing data structures
ones_array = np.ones((2, 2))    # Common in neural network initialization
random_array = np.random.rand(2, 3)  # Random values for simulations or initialization

print(f"1D Array: {array_1d}")
print(f"2D Array:\n{array_2d}")
print(f"Zeros Array:\n{zeros_array}")
print(f"Ones Array:\n{ones_array}")
print(f"Random Array:\n{random_array}")

# Array operations - vectorized operations are much faster than loops
print("\nArray Operations:")
print(f"Array + 5: {array_1d + 5}")          # Broadcasting - applying operation to each element
print(f"Array * 2: {array_1d * 2}")          # Element-wise multiplication
print(f"Array squared: {array_1d ** 2}")     # Element-wise power
print(f"Mean: {np.mean(array_1d)}")          # Statistical function
print(f"Sum: {np.sum(array_1d)}")            # Aggregate function
print(f"Standard Deviation: {np.std(array_1d)}")  # Important for data normalization

# Shape manipulation - essential for preparing data for ML algorithms
print("\nShape Manipulation:")
reshaped = array_1d.reshape(5, 1)  # Converting to column vector
print(f"Reshaped array:\n{reshaped}")
print()

# PART B: PANDAS - DATA ANALYSIS AND PREPARATION
print("\nPART B: PANDAS - DATA ANALYSIS AND PREPARATION")
print("-" * 50)

import pandas as pd

# Creating a DataFrame - the primary pandas data structure
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 45],
    'City': ['New York', 'Boston', 'Chicago', 'Denver', 'Seattle'],
    'Salary': [60000, 70000, 80000, 90000, 100000]
}

df = pd.DataFrame(data)
print("DataFrame - Sample Dataset:")
print(df)

# Basic DataFrame operations - essential for data exploration
print("\nData Exploration Operations:")
print("First 2 rows (head):")
print(df.head(2))
print("\nDataFrame Info (types and non-null values):")
print(df.info())
print("\nDataFrame Statistics (describe):")
print(df.describe())  # Key statistical summary - crucial for understanding data

# Data selection and filtering - fundamental for feature selection
print("\nData Selection and Filtering:")
print("Selecting 'Age' column (feature):")
print(df['Age'])
print("\nFiltering: rows where Age > 30 (subsetting):")
print(df[df['Age'] > 30])

# Feature engineering - creating new features
df['Experience'] = [3, 5, 8, 12, 15]  # Adding a calculated feature
print("\nFeature Engineering - Added 'Experience' column:")
print(df)
print()

#############################
# SECTION 3: DATA VISUALIZATION
#############################

print("SECTION 3: DATA VISUALIZATION")
print("=" * 60)
print("Visualization helps understand data patterns and model performance")
print("-" * 60)

import matplotlib.pyplot as plt

print("Creating visualization types commonly used in ML")
# Line plot - for showing trends and model predictions
plt.figure(figsize=(12, 8))  # Larger figure for better visibility
plt.subplot(2, 2, 1)
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, 'b-', label='sin(x)')
plt.title('Line Plot - Used for Trends & Time Series')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Scatter plot - key for understanding feature relationships
plt.subplot(2, 2, 2)
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
plt.scatter(x, y, c=colors, alpha=0.7, s=100)
plt.title('Scatter Plot - Feature Relationships')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.grid(True, linestyle='--', alpha=0.7)

# Bar plot - for categorical comparisons
plt.subplot(2, 2, 3)
categories = ['Class A', 'Class B', 'Class C', 'Class D']
values = [10, 25, 15, 30]
plt.bar(categories, values, color='maroon', alpha=0.7)
plt.title('Bar Plot - Category Comparison')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Histogram - for distribution analysis
plt.subplot(2, 2, 4)
data = np.random.randn(1000)  # Normal distribution
plt.hist(data, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title('Histogram - Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
# We'll show all plots at the end

#############################
# SECTION 4: MACHINE LEARNING FUNDAMENTALS
#############################

print("SECTION 4: MACHINE LEARNING FUNDAMENTALS")
print("=" * 60)
print("Core ML concepts and workflow")
print("-" * 60)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Key Machine Learning Concepts:")
print("1. Supervised Learning: Learning from labeled data")
print("   - Regression: Predicting continuous values (e.g., house prices)")
print("   - Classification: Predicting categories (e.g., spam detection)")
print("2. Unsupervised Learning: Finding patterns in unlabeled data")
print("   - Clustering: Grouping similar data points (e.g., customer segments)")
print("   - Dimensionality Reduction: Simplifying data while preserving information")
print("3. The ML Workflow:")
print("   - Data Collection → Data Cleaning → Feature Engineering → ")
print("     Model Selection → Training → Evaluation → Deployment")

# Generate synthetic data for our ML example
print("\nGenerating synthetic data for regression example:")
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # Feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Target = 4 + 3X + noise

print(f"Features shape: {X.shape} - Each sample has 1 feature")
print(f"Target shape: {y.shape} - Each sample has 1 target value")

# Create a DataFrame for better visualization
data_df = pd.DataFrame(np.hstack((X, y)), columns=['Feature', 'Target'])
print("\nFirst 5 rows of our synthetic dataset:")
print(data_df.head())

# Split data into training and testing sets - a crucial ML practice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split for model validation:")
print(f"Training set: {X_train.shape[0]} samples (80%)")
print(f"Testing set: {X_test.shape[0]} samples (20%)")

#############################
# SECTION 5: BUILDING ML MODELS
#############################

print("\nSECTION 5: BUILDING ML MODELS")
print("=" * 60)
print("Implementing a Linear Regression model step by step")
print("-" * 60)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed!")

# Understanding model parameters
print("\nModel Parameters (what the model learned):")
print(f"Slope (Coefficient): {model.coef_[0][0]:.3f}")  # Should be close to 3
print(f"Intercept: {model.intercept_[0]:.3f}")  # Should be close to 4
print(f"Formula: y = {model.intercept_[0]:.3f} + {model.coef_[0][0]:.3f}X")

# Make predictions on test data
y_pred = model.predict(X_test)
print("\nPredictions made on test data")

# Evaluate the model with metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error: {mse:.3f} (lower is better)")
print(f"R² Score: {r2:.3f} (1.0 is perfect fit)")

# Create final visualization
plt.figure(figsize=(10, 6))

# Plot the original data
plt.scatter(X_test, y_test, color='blue', alpha=0.7, label='Actual Data')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')

# Add annotations to make the visualization more educational
plt.title('Linear Regression Model: Predictions vs Actual')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Add a text box explaining the model
textstr = f'Model: y = {model.intercept_[0]:.2f} + {model.coef_[0][0]:.2f}X\nR² = {r2:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=10, verticalalignment='top', bbox=props)

#############################
# SECTION 6: NEXT STEPS
#############################

print("\nSECTION 6: NEXT STEPS")
print("=" * 60)
print("Suggested practice exercises and learning path")
print("-" * 60)

print("Practice Exercises:")
print("1. Try implementing a Multiple Linear Regression with more features")
print("2. Explore other algorithms like Decision Trees or k-NN")
print("3. Apply these techniques to real-world datasets from Kaggle")
print("4. Learn about data preprocessing techniques for handling missing values")
print("5. Experiment with hyperparameter tuning to improve model performance")

print("\nLearning Path Progression:")
print("→ Model Evaluation Techniques")
print("→ Classification Models")
print("→ Cross-Validation")
print("→ Feature Selection")
print("→ Regularization Techniques")
print("→ Ensemble Methods")
print("→ Introduction to Deep Learning")

# Show all plots
plt.show()

print("\nTutorial complete! You've learned the essential Python skills for machine learning.")