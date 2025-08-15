import os
import pandas as pd
import shutil

# 设置路径
base_dir = ""
csv_path = os.path.join(base_dir, "CEdata_train_target.csv")

# 加载标签文件
df = pd.read_csv(csv_path, header=0, names=["file-label", "0-label"])

# 创建分类文件夹
target_dirs = {
    1: os.path.join(base_dir, "A→B"),
    -1: os.path.join(base_dir, "B→A"),
    0: os.path.join(base_dir, "Other"),
}

# 确保目标文件夹存在
for path in target_dirs.values():
    os.makedirs(path, exist_ok=True)

# 遍历每行，移动对应文件
for file, row in df.iterrows():
    file_label = row["file-label"]
    zero_label = row["0-label"]

    src_path = os.path.join(base_dir, f'{file}.txt')
    dst_dir = target_dirs.get(file_label)

    if dst_dir and os.path.isfile(src_path):
        dst_path = os.path.join(dst_dir, src_path)
        shutil.copy2(src_path, dst_path)  # 或 shutil.move 进行移动
    else:
        print(f"跳过: 文件不存在或标签无效 → {file_label}, label={zero_label}")
