import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import read_data as rd
from tqdm import tqdm

# Блок кодирования (Encoder block)
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x_pooled = self.pool(x)
        return x, x_pooled

# Блок декодирования (Decoder block)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# U-Net
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Кодирователь
        self.enc1 = EncoderBlock(3, 64)  # Для входа с 3 каналами (RGB)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Боттлнек
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Декодирователь
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # Финальная свертка
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # Один выходной канал для бинарной маски

    def forward(self, x):
        # Кодирование
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        # Боттлнек
        x = self.bottleneck(x)

        # Декодирование
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # Финальная свертка
        x = self.final_conv(x)
        return torch.sigmoid(x)  # Для получения вероятностей (от 0 до 1)

# 1. Создание кастомного Dataset для загрузки данных
class ImageMaskDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images  # Numpy массив изображений (N, 3, H, W)
        self.masks = masks    # Numpy массив масок (N, 1, H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        return image, mask

# 2. Функция для расчета метрик
def calculate_metrics(pred, target, threshold=0.5):
    """Расчет метрик IoU и точности"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection / union).item() if union > 0 else 0.0
    accuracy = (pred == target).float().mean().item()
    return accuracy, iou


# 3. Алгоритм обучения модели
def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    best_iou = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # ---- Тренировка ----
        model.train()
        train_loss = 0.0
        train_iou, train_accuracy = 0.0, 0.0
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)

            # Прямой проход
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Обновление статистики
            train_loss += loss.item()
            acc, iou = calculate_metrics(outputs, masks)
            train_accuracy += acc
            train_iou += iou

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_iou /= len(train_loader)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, IoU: {train_iou:.4f}")

        # ---- Валидация ----
        model.eval()
        val_loss = 0.0
        val_iou, val_accuracy = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                # Прямой проход
                outputs = model(images)
                loss = criterion(outputs, masks)

                # Обновление статистики
                val_loss += loss.item()
                acc, iou = calculate_metrics(outputs, masks)
                val_accuracy += acc
                val_iou += iou

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_iou /= len(val_loader)
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, IoU: {val_iou:.4f}")

        # Сохранение лучшей модели
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")


# 4. Загрузка данных и запуск обучения
if __name__ == "__main__":
    x_data, y_data = rd.read()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.05)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    # Создание DataLoader'ов

    train_dataset = ImageMaskDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_dataset = ImageMaskDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Задание устройства (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Инициализация модели, функции потерь и оптимизатора
    model = UNet().to(device)
    criterion = nn.BCELoss()  # Бинарная кросс-энтропия для сегментации
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Запуск обучения
    epochs = 20  # Количество эпох
    train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device)

    print("Обучение завершено!")