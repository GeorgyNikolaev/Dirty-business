import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.api.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from keras.api.models import Model
from keras.api.optimizers import Adam
from keras.api.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import segmentation_models as sm
import read_data as rd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Путь к модели
model_path = 'unet_model.h5'

# Гиперпараметры
learning_rate = 0.001
epochs = 20
batch_size = 64

# Загрузка данных и разделение на тренировочную и обучающую выборки
data = rd.read()
x_train, y_train, x_test, y_test = rd.get_metric(data)


# Кастомная U-Net модель
def unet_custom(input_shape: tuple, learning_rate: float): # Замените на размер ваших изображений
    inputs = Input(input_shape)

    # Кодировщик
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # Можно добавить еще один уровень, если нужно

    # Декодер
    up5 = UpSampling2D(size=(2, 2))(conv4)
    merge5 = concatenate([up5, conv3], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = concatenate([up6, conv2], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = concatenate([up7, conv1], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv7) # Бинарная сегментация

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Предобученная U-Net модель
def unet(input_shape: tuple):
    model = sm.Unet(backbone_name='resnet34',
                    encoder_weights='imagenet',
                    classes=1,
                    activation='sigmoid',
                    input_shape=input_shape)
    return model


model = unet_custom(input_shape=(256, 256, 3), learning_rate=learning_rate)

# Аугментация
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

# Обучение модели
checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_split=0.15,
    epochs=epochs,
    callbacks=[checkpoint, reduce_lr, early_stopping]
)

# Оценка модели
score = model.evaluate(x_test, y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Mean IoU:', score[2])