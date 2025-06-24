# keras
#precision_recall_fscore_support sklearn.metrics
#code https://github.com/a18499/KubAnomaly_DataSet/blob/master/KubAnomaly_Paper.py#L14
#code https://de.wikipedia.org/wiki/Keras https://keras.io/examples/vision/mnist_convnet/

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
print(keras.__version__)

# import numpy as np
# from keras import layers
# # Model / data parameters
# num_classes = 10
# input_shape = (28, 28, 1)

# # Load the data and split it between train and test sets
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# # Scale images to the [0, 1] range
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255
# # Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")


# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# model = keras.Sequential(
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )

# model.summary()

# batch_size = 128
# epochs = 15

# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# score = model.evaluate(x_test, y_test, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])




from keras.layers import Dense
from keras.models import Sequential

# Numpy laden und festlegen des Zufalls-Startwertes
import numpy as np
np.random.seed(1337)

# Matplotlib zur grafischen Darstellung laden
import matplotlib.pyplot as plt

# Daten in Arrays speichern
eingangswerte = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
ausgangswerte = np.array([[1], [0], [0], [1]])

# Erstellt das Model mit 2 Eingangsnodes, 2 Mittelnodes und einer Ausgangsnode
num_inner = 2

model = Sequential()
model.add(Dense(num_inner, input_dim=2, activation='sigmoid'))
model.add(Dense(1))

# Kompiliert das Model, damit es spaeter verwendet werden kann
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# Trainiert das Model mit den Eingangs-
# und den entsprechenden Ausgangswerten fuer 10000 Epochen
model.fit(x=eingangswerte, y=ausgangswerte, epochs=10000, verbose=0)

# Testet die Eingangsdaten und schreibt die Ergebnisse in die Konsole
print(model.predict(eingangswerte))

# Bereitet die grafische Ausgabe mittels contourf vor
# und rastert die Eingabewerte fuer das Modell
x = np.linspace(-0.25, 1.25, 100)
(X1_raster, X2_raster) = np.meshgrid(x, x)
X1_vektor = X1_raster.flatten()
X2_vektor = X2_raster.flatten()

# Nutzt die gerasterten Eingabewerte und erzeugt Ausgabewerte
eingangswerte_grafik = np.vstack((X1_vektor, X2_vektor)).T
ausgangswerte_grafik = model.predict(eingangswerte_grafik).reshape(X1_raster.shape)

# Fragt die Gewichte der Verbindungen und die Bias-Daten ab
(gewichte, bias) = model.layers[0].get_weights()

# Contourplot der gerasterten Ausgangswerte in leicht vergroessertem
# Bereich und Legende
plt.contourf(X1_raster, X2_raster, ausgangswerte_grafik, 100)
plt.xlim(-0.25, 1.25)
plt.ylim(-0.25, 1.25)
plt.xlabel("Eingabewert $x_1$")
plt.ylabel("Eingabewert $x_2$")
plt.colorbar()

# Eintragen der Eingangsdaten in die Grafik
plt.scatter(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), color="red")

# Plot der Klassifizierungs-"Begrenzungslinien" der Aktivierungsfunktionen
for i in range(num_inner):
    plt.plot(x,
             -gewichte[0, i]/gewichte[1, i]*x
             - bias[i]/gewichte[1, i], color="black")
plt.show()