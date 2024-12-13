import numpy as np
from cost_functions import cost_functions  # Import the cost function dictionary
from activation_functions import activation_functions  # Import the activation functions dictionary
from rna import NeuralNetwork
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo



# Carregar o dataset do zoologico
zoo = fetch_ucirepo(id=111)

# Selecionar variáveis independentes (X) e dependentes (y)
X = zoo.data.features
y = zoo.data.targets

# Aplicando One-Hot Encoding
X = pandas.get_dummies(X, columns=['legs'], prefix='legs', dtype=int)
y_onehot = pandas.get_dummies(y, columns=['type'], prefix='type', dtype=int)
print("Shape of y_onehot:", y_onehot.shape)  # Debugging line to check shape

X = X.to_numpy()
y_onehot = y_onehot.to_numpy()

# Dividindo o dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)




# Define the number of input parameters dynamically
number_initial_parameters = X_train.shape[1]

# Define the network architecture
hidden_layers = [50] * 2  # 2 hidden layers with 20 neurons each
output_neurons = y_train.shape[1]  # Ensure this matches the number of classes
print("Number of output neurons:", output_neurons)  # Debugging line to check output neurons

# Combine layers into a single list
layers = hidden_layers + [output_neurons]

# Define activation functions and their derivatives
activation_funcs = [activation_functions['relu'][0]] * len(hidden_layers) + [activation_functions['softmax'][0]]  # Using ReLU for hidden layers and Sigmoid for output
activation_derivatives = [activation_functions['relu'][1]] * len(hidden_layers) + [activation_functions['softmax'][1]]  # Corresponding derivatives

# Define the cost function (assuming a mean squared error function is defined)
cost_function = cost_functions['categorical']  # Use the derivative for training

# Create an instance of the NeuralNetwork
nn = NeuralNetwork(layers, activation_funcs, activation_derivatives, number_initial_parameters, cost_function)

# Train the network and store the training and test errors
learning_rate = 0.01
epochs = 500

# After preparing your data
nn.train(X_train, X_test, y_train, y_test, epochs, learning_rate)