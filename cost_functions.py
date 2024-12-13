import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    return (2 * (y_pred - y_true)) / y_true.size

def cross_entropy_cost(A, Y):
    m = Y.shape[1]  # Number of samples
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    return cost

def cross_entropy_cost_derivative(A, Y):
    m = Y.shape[1]  # Number of samples
    return A - Y  # This is the gradient for binary cross-entropy and sigmoid combination

def categorical_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def categorical_crossentropy_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / y_true.shape[0]


# Create a dictionary mapping activation functions to their derivatives
cost_functions = {
    'MSE': (mean_squared_error, mean_squared_error_derivative),
    'cross_entropy': (cross_entropy_cost, cross_entropy_cost_derivative),
    'categorical': (categorical_crossentropy, categorical_crossentropy_derivative)
}