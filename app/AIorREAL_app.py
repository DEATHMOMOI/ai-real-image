import tkinter as tk
from tkinterdnd2 import *
from tkinter import filedialog
import pyautogui as pag
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageTk

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # 2クラス分類

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# データの前処理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# モデルを再初期化
model = SimpleCNN()
model.load_state_dict(torch.load('torch_model88.pth'))
model.eval()

def new_window():
    global window
    window = tk.Toplevel(root)
    window.wm_attributes("-alpha", 0.9)
    print(window.winfo_rootx())
    window.geometry("300x300")

def load_and_predict():
    result_text.config(text="≪ここに予測結果が表示されます≫")
    # 画像ファイルを選択
    file_path = filedialog.askopenfilename(title="画像を選択", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    img = Image.open(file_path)
    img = img.resize((250, 250))
    tk_img = ImageTk.PhotoImage(img)
    image_txt.configure(image=tk_img)
    image_txt.image = tk_img

    # 画像の読み込みと変換
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # バッチ次元を追加

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
        # 予測結果の表示
        result = "この画像は「リアル画像」" if predicted.item() == 1 else "この画像は「生成AIによる画像」"
        result_text.config(text=result)
        print(f"Prediction: {predicted.item()}")

def scs_and_predict():
    point_x = window.winfo_x() + 8
    point_y = window.winfo_y() + 5
    point_x2 = window.winfo_width()
    point_y2 = window.winfo_height()

    result_text.config(text="≪ここに予測結果が表示されます≫")
    window.destroy()

    # 画像ファイルを選択
    file_path = pag.screenshot("check.png", region=(point_x, point_y, point_x2, point_y2))
    img = Image.open("check.png")
    img = img.resize((250, 250))
    tk_img = ImageTk.PhotoImage(img)
    image_txt.configure(image=tk_img)
    image_txt.image = tk_img

    # 画像の読み込みと変換
    image = Image.open("check.png").convert('RGB')
    image = transform(image).unsqueeze(0)  # バッチ次元を追加

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
        # 予測結果の表示
        result = "この画像は「リアル画像」" if predicted.item() == 1 else "この画像は「生成AIによる画像」"
        result_text.config(text=result)
        print(f"Prediction: {predicted.item()}")

def funcDragAndDrop(event):
    # ファイル名にスペースや日本語が含まれていると{$path}で返却されることに注意
    drop_path = event.data
    drop_path = drop_path.replace("{", "").replace("}", "")
    print(event.data)
    result_text.config(text="≪ここに予測結果が表示されます≫")
    # 画像ファイルを選択
    img = Image.open(drop_path)
    img = img.resize((250, 250))
    tk_img = ImageTk.PhotoImage(img)
    image_txt.configure(image=tk_img)
    image_txt.image = tk_img

    # 画像の読み込みと変換
    image = Image.open(drop_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # バッチ次元を追加

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
        # 予測結果の表示
        result = "この画像は「リアル画像」" if predicted.item() == 1 else "この画像は「生成AIによる画像」"
        result_text.config(text=result)
        print(f"Prediction: {predicted.item()}")

def window_TF():
    if button1.config('relief')[-1] == 'sunken':
        root.attributes("-topmost", False)
        button1.config(relief=tk.RAISED)
    else:
        root.attributes("-topmost", True)
        button1.config(relief=tk.SUNKEN)

root = TkinterDnD.Tk()
root.geometry("300x435")
root.title("画像予測アプリ")

button1 = tk.Button(root, text="📌",font=(None, 8), command=window_TF)
button1.place(width=20,height=20,x=5,y=5)

# ボタンを作成してクリックで画像を選択
button = tk.Button(root, text="画像を選択して予測", command=load_and_predict)
button.place(anchor=tk.CENTER,width=200,relx=0.5,y=25)

button2 = tk.Button(root, text="範囲を選択", command=new_window)
button2.place(anchor=tk.CENTER,width=100,relx=0.25,y=55)
button3 = tk.Button(root, text="撮影して予測", command=scs_and_predict)
button3.place(anchor=tk.CENTER,width=100,relx=0.75,y=55)

labelFrame = tk.LabelFrame(width=280, height=35, text="画像をドラッグ&ドロップ", labelanchor="n")
labelFrame.drop_target_register(DND_FILES)
labelFrame.dnd_bind('<<Drop>>', funcDragAndDrop)
labelFrame.place(anchor=tk.CENTER,relx=0.5,y=90)

result_text = tk.Label(root, text="≪ここに予測結果が表示されます≫")
result_text.place(anchor=tk.CENTER,relx=0.5,y=125)

attention_text = tk.Label(root, text="この予測が必ず正しいとは限りません。\n参考にする程度で利用することをお勧めします。", font=(None, 8))
attention_text.place(anchor=tk.CENTER,relx=0.5,y=155)

image_txt = tk.Label(root)
image_txt.place(anchor=tk.CENTER,relx=0.5,y=300)

root.mainloop()
