import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# データセットのディレクトリパス
test_dir = 'test'

# 学習済みモデルの読み込み
model = load_model("model80.h5")

# テストデータの準備
test_datagen = ImageDataGenerator(rescale=1.0/255.)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # シャッフルはしない
)

# モデルの評価
evaluation = model.evaluate(test_generator)
print(f"Test Loss: {evaluation[0]}")
print(f"Test Accuracy: {evaluation[1]}")

# 予測結果の取得
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# 正解ラベルの取得
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
print(true_classes)

# 正答率を表示
correct = np.sum(predicted_classes == true_classes)
accuracy = correct / len(true_classes) * 100
print(predicted_classes)
print(f"Correct predictions: {correct} / {len(true_classes)}")
print(f"Accuracy: {accuracy:.2f}%")

# 適合率と再現率の計算
# 1（正例）のみを対象
true_positives = np.sum((true_classes == 1) & (predicted_classes == 1))
false_positives = np.sum((true_classes == 0) & (predicted_classes == 1))
false_negatives = np.sum((true_classes == 1) & (predicted_classes == 0))

# 適合率（Precision）
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

# 再現率（Recall）
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")