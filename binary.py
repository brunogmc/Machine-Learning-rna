import numpy as np
from cost_functions import cost_functions  # Import the cost function dictionary
from activation_functions import activation_functions  # Import the activation functions dictionary
from rna import NeuralNetwork
import pandas
import kagglehub
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo # Changed from RNA to rna



# Função para normalizar os dados
def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# Carregar o dataset do câncer
cancer_dataset_path = kagglehub.dataset_download("yasserh/breast-cancer-dataset")
cancer_dataset = os.listdir(cancer_dataset_path)[0]
cancer_csv = pandas.read_csv(os.path.join(cancer_dataset_path, cancer_dataset))

# Preparar os dados
df = cancer_csv.drop(columns=["id"])  # Remover coluna irrelevante
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  # Mapear M/B para 1/0

# Selecionar variáveis independentes (X) e dependentes (y)
X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values.reshape(-1, 1)

# Dividindo o dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preparação dos dados (treinamento)
X_train_normalized = normalize(X_train)
y_train_normalized = y_train

# Preparação dos dados (teste)
X_test_normalized = normalize(X_test)
y_test_normalized = y_test

# Define the number of input parameters dynamically
number_initial_parameters = X_train.shape[1]

# Define the network architecture
hidden_layers = [50] * 2  # 2 hidden layers with 20 neurons each
output_neurons = 1  # Assuming a single output for simplicity

# Combine layers into a single list
layers = hidden_layers + [output_neurons]

# Define activation functions and their derivatives
activation_funcs = [activation_functions['relu'][0]] * 2 + [activation_functions['sigmoid'][0]]  # Using ReLU for hidden layers and Sigmoid for output
activation_derivatives = [activation_functions['relu'][1]] * 2 + [activation_functions['sigmoid'][1]]  # Corresponding derivatives

# Define the cost function (assuming a mean squared error function is defined)
cost_function = cost_functions['cross_entropy']  # Use the derivative for training

# Create an instance of the NeuralNetwork
nn = NeuralNetwork(layers, activation_funcs, activation_derivatives, number_initial_parameters, cost_function)

# Train the network and store the training and test errors
learning_rate = 0.05
epochs = 1000

# After preparing your data
nn.train(X_train_normalized, y_train_normalized, X_test_normalized, y_test_normalized, epochs, learning_rate)