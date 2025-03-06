"""
Housing Price Analysis and Prediction
====================================
A comprehensive demonstration of data analysis, visualization,
and machine learning modeling using the housing dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

#############################
# 1. DATA LOADING AND EXPLORATION
#############################
print("1. DATA LOADING AND EXPLORATION")
print("-" * 50)

# Load the dataset
df = pd.read_csv('dataset.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe().round(2))

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values:")
print(missing_values)

#############################
# 2. DATA VISUALIZATION
#############################
print("\n2. DATA VISUALIZATION")
print("-" * 50)

# Create a figure with subplots
plt.figure(figsize=(15, 12))

# Plot 1: Correlation Heatmap
plt.subplot(2, 2, 1)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=14)

# Plot 2: Price vs Square Feet
plt.subplot(2, 2, 2)
sns.scatterplot(x='square_feet', y='price', data=df, hue='has_pool', palette={0:'blue', 1:'red'})
plt.title('Price vs Square Feet', fontsize=14)
plt.xlabel('Square Feet')
plt.ylabel('Price ($)')
plt.legend(title='Has Pool', labels=['No', 'Yes'])

# Plot 3: Price Distribution
plt.subplot(2, 2, 3)
sns.histplot(df['price'], kde=True)
plt.title('Price Distribution', fontsize=14)
plt.xlabel('Price ($)')
plt.ylabel('Count')

# Plot 4: Price vs Bedrooms with Age as Size
plt.subplot(2, 2, 4)
bedrooms = df['bedrooms'].unique()
colors = sns.color_palette("viridis", len(bedrooms))

for i, bedroom in enumerate(sorted(bedrooms)):
    subset = df[df['bedrooms'] == bedroom]
    plt.scatter(subset['age_years'], subset['price'], 
                s=subset['square_feet']/30, 
                color=colors[i], 
                alpha=0.7,
                label=f'{bedroom} bedrooms')

plt.title('Price vs Age by Bedrooms Count', fontsize=14)
plt.xlabel('Age (years)')
plt.ylabel('Price ($)')
plt.legend(title='Bedrooms')

plt.tight_layout()
plt.savefig('housing_visualizations.png')  # Save the visualization
print("Visualizations saved to 'housing_visualizations.png'")

#############################
# 3. FEATURE ENGINEERING AND PREPROCESSING
#############################
print("\n3. FEATURE ENGINEERING AND PREPROCESSING")
print("-" * 50)

# Create a price per square foot feature
df['price_per_sqft'] = df['price'] / df['square_feet']
print("New feature created: price_per_sqft")

# Create a feature for total rooms
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
print("New feature created: total_rooms")

# Separate features and target variable
X = df.drop(['price', 'price_per_sqft'], axis=1)
y = df['price']

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['square_feet', 'age_years', 'distance_to_downtown', 'lot_size']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("\nFeatures after preprocessing:")
print(X.head().round(2))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

#############################
# 4. MODEL TRAINING
#############################
print("\n4. MODEL TRAINING")
print("-" * 50)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Models trained:")
print("1. Linear Regression")
print("2. Random Forest Regression")

# Cross-validation for model comparison
linear_cv_scores = cross_val_score(linear_model, X, y, cv=5, scoring='r2')
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')

print("\nCross-validation R² scores (Linear Regression):")
print(f"Mean: {linear_cv_scores.mean():.4f}, Std: {linear_cv_scores.std():.4f}")
print("\nCross-validation R² scores (Random Forest):")
print(f"Mean: {rf_cv_scores.mean():.4f}, Std: {rf_cv_scores.std():.4f}")

#############################
# 5. MODEL EVALUATION
#############################
print("\n5. MODEL EVALUATION")
print("-" * 50)

# Predictions on test set
linear_pred = linear_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Calculate metrics
linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("Linear Regression Performance:")
print(f"Mean Squared Error: ${linear_mse:.2f}")
print(f"R² Score: {linear_r2:.4f}")
print(f"Root Mean Squared Error: ${np.sqrt(linear_mse):.2f}")

print("\nRandom Forest Performance:")
print(f"Mean Squared Error: ${rf_mse:.2f}")
print(f"R² Score: {rf_r2:.4f}")
print(f"Root Mean Squared Error: ${np.sqrt(rf_mse):.2f}")

# Plot predicted vs actual values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, linear_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Linear Regression: Actual vs Predicted')

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Random Forest: Actual vs Predicted')

plt.tight_layout()
plt.savefig('prediction_performance.png')
print("Performance visualization saved to 'prediction_performance.png'")

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance visualization saved to 'feature_importance.png'")

#############################
# 6. PREDICTION ON NEW DATA
#############################
print("\n6. PREDICTION ON NEW DATA")
print("-" * 50)

# Create sample data for a new house
new_house = pd.DataFrame({
    'square_feet': [2200],
    'bedrooms': [4],
    'bathrooms': [2.5],
    'age_years': [12],
    'distance_to_downtown': [6.5],
    'garage_spaces': [2],
    'lot_size': [0.23],
    'has_pool': [0],
    'total_rooms': [6.5]  # 4 + 2.5
})
# Scale the numerical features
new_house[numerical_features] = scaler.transform(new_house[numerical_features])

# Make predictions
linear_new_pred = linear_model.predict(new_house)[0]
rf_new_pred = rf_model.predict(new_house)[0]

print("New House Details:")
print("- 2,200 square feet")
print("- 4 bedrooms, 2.5 bathrooms")
print("- 12 years old")
print("- 6.5 miles from downtown")
print("- 2 garage spaces")
print("- 0.23 acre lot size")
print("- No pool")

print("\nPrice Predictions:")
print(f"Linear Regression: ${linear_new_pred:.2f}")
print(f"Random Forest: ${rf_new_pred:.2f}")
print("\nAnalysis Complete!")