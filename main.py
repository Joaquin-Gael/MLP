# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definición de una clase para la red neuronal simple
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size, learning_rate=0.1):
        """
        Constructor de la red neuronal.
        Inicializa los pesos y sesgos, y establece la tasa de aprendizaje.
        
        :param input_size: Número de características de entrada.
        :param hidden_layer_size: Número de neuronas en la capa oculta.
        :param output_size: Número de neuronas en la capa de salida.
        :param learning_rate: Tasa de aprendizaje para la actualización de pesos.
        """
        # Inicializar los pesos y sesgos de la primera capa (entrada a oculta)
        self.weights_input_to_hidden = np.random.randn(input_size, hidden_layer_size)
        self.bias_hidden_layer = np.zeros((1, hidden_layer_size))

        # Inicializar los pesos y sesgos de la segunda capa (oculta a salida)
        self.weights_hidden_to_output = np.random.randn(hidden_layer_size, output_size)
        self.bias_output_layer = np.zeros((1, output_size))

        # Establecer la tasa de aprendizaje
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """
        Función de activación sigmoide.
        
        :param x: Valor de entrada.
        :return: Salida sigmoide.
        """
        # Aplicar la función sigmoide a la entrada
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivada de la función sigmoide, utilizada en la retropropagación.
        
        :param x: Valor de entrada.
        :return: Derivada de la salida sigmoide.
        """
        # Calcular la derivada de la función sigmoide
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward_propagation(self, input_features):
        """
        Realiza la propagación hacia adelante en la red neuronal.
        
        :param input_features: Conjunto de características de entrada.
        :return: Salida de la red neuronal (predicción).
        """
        # Cálculo de la activación de la primera capa
        self.hidden_layer_activation = np.dot(input_features, self.weights_input_to_hidden) + self.bias_hidden_layer
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_activation)

        # Cálculo de la activación de la segunda capa
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_to_output) + self.bias_output_layer
        predicted_output = self.sigmoid(self.output_layer_activation)

        # Retorna la predicción final
        return predicted_output

    def backward_propagation(self, input_features, true_labels, predicted_output):
        """
        Realiza la retropropagación para actualizar los pesos y sesgos de la red neuronal.
        
        :param input_features: Conjunto de características de entrada.
        :param true_labels: Etiquetas verdaderas.
        :param predicted_output: Predicciones realizadas por la red neuronal.
        """
        # Calcular el error de predicción
        error = predicted_output - true_labels

        # Calcular las gradientes para los pesos y sesgos de la capa de salida
        d_weights_hidden_to_output = np.dot(self.hidden_layer_output.T, error * self.sigmoid_derivative(self.output_layer_activation))
        d_bias_output_layer = np.sum(error * self.sigmoid_derivative(self.output_layer_activation), axis=0, keepdims=True)

        # Calcular las gradientes para los pesos y sesgos de la capa oculta
        d_weights_input_to_hidden = np.dot(input_features.T, np.dot(error * self.sigmoid_derivative(self.output_layer_activation), self.weights_hidden_to_output.T) * self.sigmoid_derivative(self.hidden_layer_activation))
        d_bias_hidden_layer = np.sum(np.dot(error * self.sigmoid_derivative(self.output_layer_activation), self.weights_hidden_to_output.T) * self.sigmoid_derivative(self.hidden_layer_activation), axis=0)

        # Actualizar pesos y sesgos de la capa de salida
        self.weights_hidden_to_output -= self.learning_rate * d_weights_hidden_to_output
        self.bias_output_layer -= self.learning_rate * d_bias_output_layer

        # Actualizar pesos y sesgos de la capa oculta
        self.weights_input_to_hidden -= self.learning_rate * d_weights_input_to_hidden
        self.bias_hidden_layer -= self.learning_rate * d_bias_hidden_layer

    def train(self, input_features, true_labels, epochs=10000):
        """
        Entrena la red neuronal utilizando los datos de entrenamiento.
        
        :param input_features: Conjunto de características de entrada para entrenamiento.
        :param true_labels: Etiquetas verdaderas para el entrenamiento.
        :param epochs: Número de iteraciones para el entrenamiento.
        """
        # Ciclo de entrenamiento para un número definido de épocas
        for epoch in range(epochs):
            # Propagación hacia adelante para obtener predicciones
            predicted_output = self.forward_propagation(input_features)

            # Retropropagación para ajustar los pesos y sesgos
            self.backward_propagation(input_features, true_labels, predicted_output)

            # Imprimir la pérdida cada 1000 épocas
            if epoch % 1000 == 0:
                loss = np.mean(np.square(predicted_output - true_labels))
                print(f"Epoch {epoch} - Pérdida: {loss}")

    def predict(self, input_features):
        """
        Realiza predicciones utilizando la red neuronal entrenada.
        
        :param input_features: Conjunto de características de entrada para realizar predicciones.
        :return: Predicciones (0 o 1).
        """
        # Propagación hacia adelante para obtener predicciones
        predicted_output = self.forward_propagation(input_features)
        
        # Redondear las predicciones para obtener 0 o 1
        return np.round(predicted_output).astype(int)

    def accuracy(self, predicted_labels, true_labels):
        """
        Calcula la exactitud de las predicciones realizadas por la red neuronal.
        
        :param predicted_labels: Predicciones realizadas.
        :param true_labels: Etiquetas verdaderas.
        :return: Porcentaje de exactitud.
        """
        # Comparar las predicciones con las etiquetas verdaderas y calcular la precisión
        return np.mean(predicted_labels == true_labels.reshape(-1, 1)) * 100

# Preparación de los datos

# Cargar el conjunto de datos Iris
iris_dataset = load_iris()

# Crear un DataFrame de pandas con los datos de Iris
iris_dataframe = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
iris_dataframe['target'] = iris_dataset.target

# Filtrar solo las clases Setosa (0) y Versicolor (1)
filtered_iris_data = iris_dataframe[iris_dataframe['target'] != 2]

# Seleccionar las características (ancho y largo del pétalo)
feature_data = filtered_iris_data[['petal length (cm)', 'petal width (cm)']].values
target_data = filtered_iris_data['target'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.2, random_state=42)

# Escalar los datos (normalización)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenamiento del modelo

# Crear una instancia de la red neuronal
input_size = X_train.shape[1]
hidden_layer_size = 4
output_size = 1
simple_nn = SimpleNeuralNetwork(input_size, hidden_layer_size, output_size)

# Entrenar la red neuronal
simple_nn.train(X_train, y_train.reshape(-1, 1), epochs=10000)

# Evaluar el modelo

# Hacer predicciones en el conjunto de prueba
y_pred = simple_nn.predict(X_test)

# Calcular la exactitud
model_accuracy = simple_nn.accuracy(y_pred, y_test)
print(f"Exactitud del modelo: {model_accuracy:.2f}%")

# Visualización de la frontera de decisión

# Crear una malla para dibujar la frontera de decisión
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Propagación hacia adelante en la malla de puntos
Z = simple_nn.forward_propagation(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Dibujar la frontera de decisión
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='x')
plt.xlabel('Largo del pétalo (normalizado)')
plt.ylabel('Ancho del pétalo (normalizado)')
plt.title('Frontera de decisión de la red neuronal')
plt.show()
