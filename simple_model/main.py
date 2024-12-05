import keras
from keras.api import layers
import read_data as rd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


learning_rate = 0.001
epochs = 20
batch_size = 64


data = rd.read()
x_train, y_train, x_test, y_test = rd.get_metric(data)


inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_outputs = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_outputs)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_outputs = layers.add([x, block_1_outputs])
x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_outputs)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_outputs = layers.add([x, block_2_outputs])
x = layers.Conv2D(64, 3, activation="relu")(block_3_outputs)
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="dirty_business")

model.compile(optimizer=keras.api.optimizers.Adam(learning_rate=learning_rate),
              loss=keras.api.losses.binary_crossentropy,
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

print(model.evaluate(x_test, y_test))

