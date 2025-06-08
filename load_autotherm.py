# step1_load_autotherm_full_with_progress.py
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os

# 打印当前目录（让你知道 CSV 会保存在哪里）
print("📁 当前目录为：", os.getcwd())

# 加载数据（含提示）
print("⏳ 正在从 Hugging Face 加载 AutoTherm 数据集...")
dataset = load_dataset("kopetri/AutoTherm")
train_data = dataset["train"]

# 转为 DataFrame（可加进度条展示大小）
print(f"✅ 数据加载完成，共有 {len(train_data)} 条记录。")

# 转换为 pandas DataFrame
print("📄 正在转换为 DataFrame...")
df = pd.DataFrame(train_data)


# 保留以下特征列
features = [
    "Age", "Gender", "Weight", "Height", "Bodyfat", "Bodytemp",
    "Sport-Last-Hour", "Time-Since-Meal", "Tiredness",
    "Clothing-Level", "Radiation-Temp", "PCE-Ambient-Temp",
    "Air-Velocity", "Metabolic-Rate", "Emotion-Self", "Emotion-ML",
    "Nose", "Neck", "RShoulder", "RElbow", "LShoulder", "LElbow",
    "REye", "LEye", "REar", "LEar",
    "Wrist_Skin_Temperature", "Heart_Rate", "GSR",
    "Ambient_Temperature", "Ambient_Humidity", "Solar_Radiation"
]
label = "Label"

print("🧪 保留特征列 + 标签列...")
df = df[features + [label]]

# 保存为 CSV（在当前路径）
save_path = "autotherm_full_features.csv"
df.to_csv(save_path, index=False)

print(f"✅ 保存完成！文件已写入：{save_path}")
print(f"📊 保存的数据大小：{df.shape[0]} 行, {df.shape[1]} 列")
