import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import read_data as rd
from tqdm import tqdm


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Кодировщик
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Декодер
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.up6 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.up7 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
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


# Функция для подсчета accuracy
def accuracy(y_pred, y_true, t):
    with torch.no_grad():
        # Преобразуем предсказания и истинные метки в бинарный формат
        y_pred = (y_pred > t).float()  # бинаризуем предсказания
        correct = (y_pred == y_true).float()  # сравниваем с истинными метками
        acc = correct.sum() / correct.numel()  # точность
    return acc.item()


# Функция для подсчета IoU
def iou(y_pred, y_true, threshold=0.5):
    with torch.no_grad():
        # Бинаризуем предсказания на основе порога
        y_pred = (y_pred > threshold).float()

        # Вычисляем пересечение и объединение
        intersection = (y_pred * y_true).sum()  # пересечение
        union = y_pred.sum() + y_true.sum() - intersection  # объединение

        # Избегаем деления на ноль
        iou_score = intersection / union if union != 0 else 0.0

    return float(iou_score)  # возвращаем как обычное число


learning_rate = 0.001
epochs = 0
batch_size = 4

# Создаем модель и загружаем веса
model = UNet()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

x_data, y_data = rd.read()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.05)
print(len(x_train))
print('Данные загружены')

# Преобразование данных в тензоры PyTorch
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)  # (batch, height, width, channels) -> (batch, channels, height, width)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Добавляем канал для масок

x_test_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)  # (batch, height, width, channels) -> (batch, channels, height, width)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Добавляем канал для масок

# Создание DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Функция потерь
criterion = nn.BCELoss()

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Перевод модели в режим обучения
model.train()
min_loss = 10**6
epochs_for_loss = 0
# Цикл обучения
for epoch in range(epochs):
    model.train()  # Устанавливаем модель в режим тренировки
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    max_iou = 0.0

    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(images)

        # Вычисляем потерю
        loss = criterion(outputs, masks)

        # Обратный проход и обновление весов
        loss.backward()
        optimizer.step()

        # Подсчитываем accuracy и IoU
        acc = accuracy(outputs, masks, 0.5)
        iou_score = iou(outputs, masks, 0.5)

        running_loss += loss.item()
        running_acc += acc
        running_iou += iou_score
        max_iou = max(max_iou, running_iou)

    # Среднее значение для одной эпохи
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    epoch_iou = running_iou / len(train_loader)
    if epoch_loss <= min_loss:
        min_loss = epoch_loss
        epochs_for_loss = 0
    elif epochs_for_loss >= 6:
        # Сохранение обученной модели
        torch.save(model.state_dict(), "unet_trained_3.pth")
        print("Обученная модель сохранена в 'unet_trained_3.pth'")

    epochs_for_loss += 1

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, IoU: {epoch_iou:.4f}, max IoU: {max_iou:.4f}")

# Сохранение обученной модели
# torch.save(model.state_dict(), "unet_trained_2.pth")
# print("Обученная модель сохранена в 'unet_trained_2.pth'")

# Загрузка архитектуры модели (обязательно определить класс UNet)
loaded_model = UNet()

# Загрузка весов в модель
loaded_model.load_state_dict(torch.load("unet_trained_3.pth"))
loaded_model.eval()
# Если доступен GPU, перемещаем модель на устройство
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

print("Модель загружена")

# Преобразование для входных изображений
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразование в тензор
])

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for x, y in test_loader:
    # x = x.squeeze().cpu().numpy()
    y = y.squeeze().cpu().numpy()
    print(np.array(x).shape)
    print(np.array(y).shape)
    print('-'*30)
    predicts = []

    image_i = []
    for i, t in enumerate(thresholds):
        # # Получаем изображение и преобразуем его в тензор
        # test_image = x
        # test_image_tensor = transform(test_image).unsqueeze(0).to(device)
        # test_image_tensor = x.to(torch.float32)

        # Выполняем предсказание
        with torch.no_grad():
            predicted_mask = model(x)

        # Преобразуем предсказанную маску
        predicted_mask = predicted_mask.squeeze().cpu().numpy()
        predicted_mask_binary = (predicted_mask > t).astype(np.uint8) * 255
        predicted_mask_binary = np.stack([predicted_mask_binary] * 3, axis=-1)
        predicts.append(predicted_mask_binary)

    # Визуализируем исходное изображение и предсказанную маску
    plt.figure(figsize=(14, 8))

    # Исходное изображение
    plt.subplot(3, 3, 1)
    plt.title(f"Исходное изображение {i + 1}")
    plt.imshow(x.squeeze().cpu().numpy().reshape((144, 256, 3)))

    # Предсказанная маска
    plt.subplot(3, 3, 2)
    plt.title(f"Верная маска")
    plt.imshow(y, cmap="gray")

    for i, t in enumerate(thresholds):
        plt.subplot(3, 3, i+3)
        plt.title(f"Предсказанная маска {i + 1}")
        plt.imshow(predicts[i], cmap="gray")

    plt.show()

