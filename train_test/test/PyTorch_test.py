import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# データの前処理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# テストデータセットの読み込み
test_dataset = datasets.ImageFolder(root='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# モデルを再初期化
model = SimpleCNN()
model.load_state_dict(torch.load('torch_model883.pth'))
model.eval()  # 評価モードに設定

# テストの実行
correct = 0
total = 0
true = []
results = []

# 真陽性、偽陽性、偽陰性のカウント
TP = 0
FP = 0
FN = 0

with torch.no_grad():  # 勾配計算を無効にする
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 最大値を持つクラスを取得

        for i in range(labels.size(0)):
            true.append(labels[i].item())
            results.append(predicted[i].item())

            if predicted[i] == labels[i]:  # 正しい予測
                correct += 1
            if predicted[i] == 1 and labels[i] == 1:  # 真陽性
                TP += 1
            if predicted[i] == 1 and labels[i] == 0:  # 偽陽性
                FP += 1
            if predicted[i] == 0 and labels[i] == 1:  # 偽陰性
                FN += 1

        total += labels.size(0)

test_accuracy = correct / total * 100

# 適合率と再現率の計算
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# 正解とモデルの予測を表示
print(true)
print(results)
print(f'Test Accuracy: {test_accuracy:.4f}% {correct}/{total}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
