import numpy as np
from cost_functions import cost_functions, mean_squared_error  # Import the cost function dictionary
from activation_functions import activation_functions  # Import the activation functions dictionary
import pandas
import kagglehub
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

class NeuralNetwork:

    def __init__(self, layers: list, layers_activators: list, layers_derivatives: list, number_initial_parameters: int, cost_function):
        self.layers = layers
        self.layers_activators = layers_activators
        self.layers_derivatives = layers_derivatives
        self.number_initial_parameters = number_initial_parameters
        self.cost_function = cost_function[0]
        self.cost_function_derivative = cost_function [1]
        self.layers_weights, self.layers_biases = self._initialize_network()
        np.random.seed(42)

    def _initialize_network(self):
        weights_list = []
        biases_list = []

        # Initialize the first layer
        weights, biases = self.initialize_neurons_layer(self.number_initial_parameters, self.layers[0])
        weights_list.append(weights)
        biases_list.append(biases)

        # Initialize subsequent layers
        for layer_number in range(1, len(self.layers)):
            weights, biases = self.initialize_neurons_layer(self.layers[layer_number - 1], self.layers[layer_number])
            weights_list.append(weights)
            biases_list.append(biases)

        return weights_list, biases_list

    def initialize_neurons_layer(self, previous_length, layer_length):
        # Initialize weights with small random values
        weights = np.random.randn(layer_length, previous_length) * 0.01
        # Initialize biases to zeros
        biases = np.zeros((layer_length, 1))  # Shape (layer_length, 1)
        return weights, biases

    def forward_propagation(self, input):
        activations = [input.T]  # Transpose input to match the shape (features, samples)
        linear_outputs = []

        A = activations[0]  # Start with the input
        for layer_index in range(len(self.layers)):
            weights = self.layers_weights[layer_index]
            biases = self.layers_biases[layer_index]

            # Compute Z
            Z = np.dot(weights, A) + biases
            linear_outputs.append(Z)

            # Apply activation function
            A = self.layers_activators[layer_index](Z)
            activations.append(A)

        return activations, linear_outputs

    def backward_propagation(self, Y, activations, linear_outputs):
        m = Y.shape[1]  # Number of samples
        gradients = {}
        L = len(self.layers)  # Total number of layers

        # Compute the derivative of the cost with respect to the activation of the last layer
        cost_derivative = self.cost_function_derivative(activations[-1], Y)

        # Initialize dA as the cost derivative
        dA = cost_derivative
        for layer_index in reversed(range(L)):
            A_prev = activations[layer_index]
            Z = linear_outputs[layer_index]
            weights = self.layers_weights[layer_index]

            # Compute dZ
            dZ = dA * self.layers_derivatives[layer_index](Z)

            # Compute gradients
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Store gradients
            gradients[f'dW{layer_index + 1}'] = dW
            gradients[f'db{layer_index + 1}'] = db

            # Update dA for the next layer
            if layer_index > 0:  # Skip for the input layer
                dA = np.dot(weights.T, dZ)

        return gradients


    def train(self, X: np.ndarray, Y: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, epochs: int, learning_rate: float):
        """
        Train the neural network using forward and backward propagation.
        """
        training_mse = []
        test_mse = []

        for epoch in range(epochs):
            # Forward propagation
            activations, linear_outputs = self.forward_propagation(X)

            # Backward propagation
            gradients = self.backward_propagation(Y.T, activations, linear_outputs)  # Transpose Y here

            # Update weights and biases
            for layer_index in range(len(self.layers)):
                self.layers_weights[layer_index] -= learning_rate * gradients[f'dW{layer_index + 1}']
                self.layers_biases[layer_index] -= learning_rate * gradients[f'db{layer_index + 1}']

            # Calculate training and test MSE
            if epoch % 10 == 0:
                train_mse = mean_squared_error(Y, activations[-1])  # Calculate MSE for training set
                test_activations, _ = self.forward_propagation(X_test)
                test_mse_value = mean_squared_error(Y_test, test_activations[-1])  # Calculate MSE for test set

                training_mse.append(train_mse)
                test_mse.append(test_mse_value)

                print(f"Epoch {epoch}, Training MSE: {train_mse:.6f}, Test MSE: {test_mse_value:.6f}")

        # Plotting the MSE
        plt.plot(range(0, epochs, 10), training_mse, label='Training MSE')
        plt.plot(range(0, epochs, 10), test_mse, label='Test MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Training and Test MSE over Epochs')
        plt.legend()
        plt.show()
    
    def predict(self, input):
        """
        Predict the output for a given input using forward propagation.
        """
        activations, _ = self.forward_propagation(input)
        return activations[-1]  # Return the output of the last layer


