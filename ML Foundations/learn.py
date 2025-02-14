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
