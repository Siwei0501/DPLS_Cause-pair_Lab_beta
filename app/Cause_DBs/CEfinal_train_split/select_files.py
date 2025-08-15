import os
import random
from typing import Literal

import numpy as np
import pandas as pd


def select(relation: Literal['AB', 'BA'], threshold: list, test_SAMPLE: int = 200, seed=5):
    np.random.seed(seed)
    random.seed(seed)
    # 文件路径
    folder_path = rf'C:\Users\19012\OneDrive\Programs\Cause_GUI\Cause_effect_pair\dataset\{relation[0]}→{relation[1]}'

    if os.path.exists(rf'temp_file'):
        pass
    else:
        os.mkdir(rf'temp_file')
        print('已创建临时文件夹 temp_file')

    # 获取当前目录下所有以 'train' 开头并以 '.txt' 结尾的文件
    files = [f for f in os.listdir(folder_path) if f.startswith('train') and f.endswith('.txt')]
    # 定义用于记录处于阈值内文件的文件名
    record_file = rf'C:\Users\19012\OneDrive\Programs\Cause_GUI\Cause_effect_pair\temp_file\{relation}_files_OUT2_of_{threshold[0]}-{threshold[1]}.txt'
    print("Reading..")

    # 读取当前阈值的文件记录（如果存在）
    if os.path.exists(record_file):

        print('检测到长度记录文件')

        with open(record_file, 'r') as f:
            OUT_threshold_files = f.read().splitlines()
            valid_files = pd.concat([pd.DataFrame(files), pd.DataFrame(OUT_threshold_files)]).drop_duplicates(
                keep=False)
            valid_files = valid_files[0].to_list()
    else:
        OUT_threshold_files = set()
        valid_files = []

        # 过滤出行数小于阈值的文件
        for _file in files:
            file_path = os.path.join(folder_path, _file)

            # 如果文件已经记录为大文件（超出阈值），则跳过
            if _file in OUT_threshold_files:
                continue

            # 读取文件并检查行数
            try:
                df = pd.read_csv(file_path, delimiter='\t')  # 假设文件是制表符分隔的
                if threshold[0] <= len(df) <= threshold[1]:
                    valid_files.append(_file)
                else:
                    # 如果文件行数大于阈值，记录文件名
                    OUT_threshold_files.add(_file)
                    with open(record_file, 'a') as f:
                        f.write(_file + '\n')
            except Exception as e:
                print(f"无法读取文件 {_file}: {e}")

    #
    if len(valid_files) < test_SAMPLE:
        test_SAMPLE = len(valid_files)
        print('检测到: 符合阈值的文件数量 < 需要测试的数量')

    # 随机选择50个文件（如果文件数小于50，选择所有文件）
    selected_files = random.sample(valid_files, min(test_SAMPLE, len(valid_files)))
    read_in_files = [pd.read_csv(os.path.join(folder_path, file), sep='\t', header=None, dtype=float) for file in
                     selected_files]
    file_names = [name_[:-4] for name_ in selected_files]

    return read_in_files, file_names
