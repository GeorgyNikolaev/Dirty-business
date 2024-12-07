import os

import keras.api.models
import numpy as np
import torch
import torch.nn as nn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.api.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from keras.api.models import Model
from keras.api.optimizers import Adam
from keras.api.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
# import segmentation_models as sm
from sklearn.model_selection import train_test_split
import read_data as rd
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Путь к модели
model_path = 'unet_model_1.keras'

# Гиперпараметры
learning_rate = 0.001
epochs = 100
batch_size = 4

# Загрузка данных и разделение на тренировочную и обучающую выборки
x_data, y_data = rd.read()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
print(len(x_data))
print('Данные загружены')

# Кастомная U-Net модель
def unet_custom(input_shape: tuple, learning_rate: float): # Замените на размер ваших изображений
    inputs = Input(input_shape)

    # Кодировщик
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # conv4 = Conv2D(128, 3, activation='relu', padding='same')(pool1) #test
    # conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4) #test

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # Можно добавить еще один уровень, если нужно

    # Декодер
    # up5 = UpSampling2D(size=(2, 2))(conv4) #test
    # merge5 = concatenate([up5, conv1], axis=3) #test
    # conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge5) #test
    # conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5) #test

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
    # outputs = Conv2D(1, 1, activation='sigmoid')(conv5) # Бинарная сегментация test

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Предобученная U-Net модель
# def unet(input_shape: tuple):
#     model = sm.Unet(backbone_name='resnet34',
#                     encoder_weights='imagenet',
#                     classes=1,
#                     activation='sigmoid',
#                     input_shape=input_shape)
#     return model


model = unet_custom(input_shape=(len(x_data[0]), len(x_data[0][0]), 3), learning_rate=learning_rate)

# Аугментация
# datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

# Обучение модели
checkpoint = ModelCheckpoint(model_path, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
print('Модель готова к обучению')

# history = model.fit(
#     # datagen.flow(x_train, y_train, batch_size=batch_size),
#     x_train, y_train,
#     batch_size=batch_size,
#     validation_split=0.15,
#     epochs=epochs,
#     callbacks=[checkpoint, reduce_lr, early_stopping]
# )
print('Модель обучилась')

# model_load = keras.api.models.load_model('unet_model_1.keras')


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Кодировщик
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Декодер
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.up6 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.up7 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Кодировщик
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)

        # Декодер
        up5 = self.up5(c4)
        merge5 = torch.cat([up5, c3], dim=1)
        c5 = self.conv5(merge5)

        up6 = self.up6(c5)
        merge6 = torch.cat([up6, c2], dim=1)
        c6 = self.conv6(merge6)

        up7 = self.up7(c6)
        merge7 = torch.cat([up7, c1], dim=1)
        c7 = self.conv7(merge7)

        output = self.final_conv(c7)
        return self.sigmoid(output)


weights = np.load('keras_weights.npy', allow_pickle=True).item()


def load_weights(pytorch_model, keras_weights):
    for name, module in pytorch_model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_weights = keras_weights.get(name)
            if layer_weights:
                # Установка весов (Keras: [H, W, C_in, C_out] -> PyTorch: [C_out, C_in, H, W])
                module.weight.data = torch.tensor(layer_weights[0]).permute(3, 2, 0, 1)
                module.bias.data = torch.tensor(layer_weights[1])
        elif isinstance(module, nn.ConvTranspose2d):
            # Аналогичная логика для транспонированных сверток
            pass


# Создаем модель и загружаем веса
model_torch = UNet()
load_weights(model_torch, weights)


# loaded_model = torch.load('unet_model.pth')
# loaded_model.eval()

print("Модкль загружена")
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for x, y in zip(x_test, y_test):
    print('-'*30)
    y_ = np.stack([y] * 3, axis=-1)
    image = np.concatenate((x, y_), axis=1)

    image_i = []
    for i, t in enumerate(thresholds):
        x_ = np.array(np.expand_dims(x, axis=0).reshape((1, 3, 256, 144)) * 255, dtype=int)
        x_ = torch.Tensor(x_)
        y_pred = model_torch(x_)
        y_pred = y_pred.detach().numpy() * 255
        y_pred = np.array(y_pred).reshape((144, 256))
        m = keras.api.metrics.BinaryIoU(threshold=t)
        m.update_state(y, y_pred)
        print('IoU при t=' + str(t) + ': ' + str(m.result().numpy()))

        binary_y_pred = np.where(y_pred > t * 255, 255, 0)
        binary_y_pred = binary_y_pred.astype(np.uint8)
        binary_y_pred = np.stack([binary_y_pred] * 3, axis=-1)
        if i == 0:
            image = np.concatenate((image, binary_y_pred), axis=1)

        elif i % 3 == 1:
            image_i = binary_y_pred
        elif i % 3 == 2:
            image_i = np.concatenate((image_i, binary_y_pred), axis=1)
        else:
            image_i = np.concatenate((image_i, binary_y_pred), axis=1)
            image = np.concatenate((image, image_i), axis=0)

    cv2.imshow('image', image)
    cv2.waitKey(0)

# Оценка модели
score = model.evaluate(x_test, y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Mean IoU:', score[2])