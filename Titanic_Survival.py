import numpy as np
import pandas as pd

# Load the dataset (assuming a CSV file with a 'Survived' column)
# Replace 'path_to_csv' with the actual path to the Titanic dataset CSV
data = pd.read_csv('Titanic-Dataset.csv')

# Preprocess the dataset
# For simplicity, let's consider 'Pclass', 'Age', 'SibSp', 'Parch', and 'Fare' as features
# and 'Survived' as the target variable. We will also convert 'Sex' to a binary variable.
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()

# Separate features and target variable
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
y = data['Survived'].values

# Normalize the feature data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Add intercept term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(X, y, weights):
    m = len(y)
    predictions = sigmoid(X @ weights)
    cost = (-1/m) * (y @ np.log(predictions) + (1 - y) @ np.log(1 - predictions))
    return cost

# Gradient descent
def gradient_descent(X, y, weights, lr, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        weights = weights - lr * (X.T @ (sigmoid(X @ weights) - y)) / m
        cost_history[i] = cost_function(X, y, weights)

    return weights, cost_history

# Initialize weights
weights = np.zeros(X.shape[1])

# Hyperparameters for the learning algorithm
lr = 0.01
iterations = 1000

# Train the model
weights, cost_history = gradient_descent(X, y, weights, lr, iterations)

# Predict function
def predict(X, weights):
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return sigmoid(X @ weights) >= 0.5

# Generate random data points and predict
random_data = np.random.rand(5, 6)
predictions = predict(random_data, weights)

# Output predictions
print("Predictions on random data points:")
for random_data, predictions in enumerate(predictions):
    print(f"Data Point {random_data}: {predictions} with {random_d}")