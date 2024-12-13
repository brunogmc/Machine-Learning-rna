import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(z):
    z = np.clip(z, -500, 500)  # Limita valores extremos p/ evitar overflow
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Para evitar overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_derivative(z):
    s = softmax(z)
    return np.diagflat(s) - np.dot(s, s.T)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)


# Create a dictionary mapping activation functions to their derivatives
activation_functions = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'softmax': (softmax, softmax_derivative),
    'linear': (linear, linear_derivative)
}



