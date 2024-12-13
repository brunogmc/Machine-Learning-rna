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

# Carregar o dataset de performance de estudantes
data_path = kagglehub.dataset_download('nikhil7280/student-performance-multiple-linear-regression')
dataset = os.listdir(data_path)[0]
data_csv = pandas.read_csv(os.path.join(data_path, dataset))

# Selecionar variáveis independentes (X) e dependentes (y)
X = data_csv[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
X = X.copy()
X['Extracurricular Activities'] = X['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
X = X.values
y = data_csv['Performance Index'].values
y = normalize(y.reshape(-1, 1))

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
hidden_layers = [50]  
output_neurons = 1  # Assuming a single output for simplicity

# Combine layers into a single list
layers = hidden_layers + [output_neurons]

# Define activation functions and their derivatives
activation_funcs = [activation_functions['relu'][0]] * len(hidden_layers) + [activation_functions['sigmoid'][0]]  # Using ReLU for hidden layers and Sigmoid for output
activation_derivatives = [activation_functions['relu'][1]] * len(hidden_layers) + [activation_functions['sigmoid'][1]]  # Corresponding derivatives

# Define the cost function (assuming a mean squared error function is defined)
cost_function = cost_functions['MSE']  # Use the derivative for training

# Create an instance of the NeuralNetwork
nn = NeuralNetwork(layers, activation_funcs, activation_derivatives, number_initial_parameters, cost_function)

# Train the network and store the training and test errors
learning_rate = 0.05
epochs = 250

# After preparing your data
nn.train(X_train_normalized, y_train_normalized, X_test_normalized, y_test_normalized, epochs, learning_rate)




# Função para calcular o coeficiente de determinação (R2) para o problema de regressão
def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Avaliação dos dados de treinamento
y_train_pred = nn.forward(X_train_normalized)
r2_train = r2_score(y_train_normalized, y_train_pred)
print(f"Coeficiente de Determinação (R2) - treinamento: {r2_train}")

# Avaliação dos dados de teste
y_test_pred = nn.forward(X_test_normalized)
r2_test = r2_score(y_test_normalized, y_test_pred)
print(f"Coeficiente de Determinação (R2) - teste: {r2_test}")
