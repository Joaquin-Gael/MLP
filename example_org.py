# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definición de una clase para la red neuronal simple
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Constructor de la red neuronal.
        Inicializa los pesos y sesgos, y establece la tasa de aprendizaje.
        
        :param input_size: Número de características de entrada.
        :param hidden_size: Número de neuronas en la capa oculta.
        :param output_size: Número de neuronas en la capa de salida.
        :param learning_rate: Tasa de aprendizaje para la actualización de pesos.
        """
        # Inicializar los pesos y sesgos de la primera capa (entrada a oculta)
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))

        # Inicializar los pesos y sesgos de la segunda capa (oculta a salida)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

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

    def ReLu(x):
        """
        ReLu devuelve el valor mayor a cero

        Args:
            x (float): Valor X

        Returns:
            float: Salida Y
        """
        # Aplicar la función ReLu para mantener valores positivos
        return np.maximum(0, x)

    def forward_propagation(self, X):
        """
        Realiza la propagación hacia adelante en la red neuronal.
        
        :param X: Conjunto de características de entrada.
        :return: Salida de la red neuronal (predicción).
        """
        # Cálculo de la activación de la primera capa
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Cálculo de la activación de la segunda capa
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        y_hat = self.sigmoid(self.z2)

        # Retorna la predicción final
        return y_hat

    def backward_propagation(self, X, y, y_hat):
        """
        Realiza la retropropagación para actualizar los pesos y sesgos de la red neuronal.
        
        :param X: Conjunto de características de entrada.
        :param y: Etiquetas verdaderas.
        :param y_hat: Predicciones realizadas por la red neuronal.
        """
        # Calcular el error de predicción
        error = y_hat - y

        # Calcular las gradientes para los pesos y sesgos de la capa de salida
        dW2 = np.dot(self.a1.T, error * self.sigmoid_derivative(self.z2))
        db2 = np.sum(error * self.sigmoid_derivative(self.z2), axis=0, keepdims=True)

        # Calcular las gradientes para los pesos y sesgos de la capa oculta
        dW1 = np.dot(X.T, np.dot(error * self.sigmoid_derivative(self.z2), self.W2.T) * self.sigmoid_derivative(self.z1))
        db1 = np.sum(np.dot(error * self.sigmoid_derivative(self.z2), self.W2.T) * self.sigmoid_derivative(self.z1), axis=0)

        # Actualizar pesos y sesgos de la capa de salida
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        # Actualizar pesos y sesgos de la capa oculta
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=10000):
        """
        Entrena la red neuronal utilizando los datos de entrenamiento.
        
        :param X: Conjunto de características de entrada para entrenamiento.
        :param y: Etiquetas verdaderas para el entrenamiento.
        :param epochs: Número de iteraciones para el entrenamiento.
        """
        # Ciclo de entrenamiento para un número definido de épocas
        for epoch in range(epochs):
            # Propagación hacia adelante para obtener predicciones
            y_hat = self.forward_propagation(X)

            # Retropropagación para ajustar los pesos y sesgos
            self.backward_propagation(X, y, y_hat)

            # Imprimir la pérdida cada 1000 épocas
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y_hat - y))
                print(f"Epoch {epoch} - Pérdida: {loss}")

    def predict(self, X):
        """
        Realiza predicciones utilizando la red neuronal entrenada.
        
        :param X: Conjunto de características de entrada para realizar predicciones.
        :return: Predicciones (0 o 1).
        """
        # Propagación hacia adelante para obtener predicciones
        y_hat = self.forward_propagation(X)
        
        # Redondear las predicciones para obtener 0 o 1
        return np.round(y_hat).astype(int)

    def accuracy(self, y_pred, y_true):
        """
        Calcula la exactitud de las predicciones realizadas por la red neuronal.
        
        :param y_pred: Predicciones realizadas.
        :param y_true: Etiquetas verdaderas.
        :return: Porcentaje de exactitud.
        """
        # Comparar las predicciones con las etiquetas verdaderas y calcular la precisión
        return np.mean(y_pred == y_true.reshape(-1, 1)) * 100

# Preparación de los datos

# Cargar el conjunto de datos Iris
iris = load_iris()

# Crear un DataFrame de pandas con los datos de Iris
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Filtrar solo las clases Setosa (0) y Versicolor (1)
df = df[df['target'] != 2]

# Seleccionar las características (ancho y largo del pétalo)
X = df[['petal length (cm)', 'petal width (cm)']].values
y = df['target'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos (normalización)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenamiento del modelo

# Crear una instancia de la red neuronal
input_size = X_train.shape[1]
hidden_size = 4
output_size = 1
nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)

# Entrenar la red neuronal
nn.train(X_train, y_train.reshape(-1, 1), epochs=10000)

# Evaluar el modelo

# Hacer predicciones en el conjunto de prueba
y_pred = nn.predict(X_test)

# Calcular la exactitud
accuracy = nn.accuracy(y_pred, y_test)
print(f"Exactitud del modelo: {accuracy:.2f}%")

# Visualización de la frontera de decisión

# Crear una malla para dibujar la frontera de decisión
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Propagación hacia adelante en la malla de puntos
Z = nn.forward_propagation(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Dibujar la frontera de decisión
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='x')
plt.xlabel('Largo del pétalo (normalizado)')
plt.ylabel('Ancho del pétalo (normalizado)')
plt.title('Frontera de decisión de la red neuronal')
plt.show()
