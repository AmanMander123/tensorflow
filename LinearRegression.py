import tensorflow as tf
import numpy as np
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(loss='mean_squared_error',optimizer='sgd')
xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
ys = np.array([0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00], dtype=float)

model.fit(xs, ys, epochs=500)
print(model.predict([7.0]))
