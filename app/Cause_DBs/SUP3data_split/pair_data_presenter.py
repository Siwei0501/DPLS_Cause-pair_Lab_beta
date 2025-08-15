import os
import random
from typing import Literal

import numpy as np
import pandas as pd



def return_cause_pair(relation: Literal['AB', 'BA', 'AB&BA'], threshold: list, test_SAMPLE: int = 200, seed=5, **kwargs):

    X = None
    y = None

    np.random.seed(seed)
    random.seed(seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if relation == 'AB&BA':
        relation = ['AB', 'BA']

    elif relation == 'AB':
        relation = ['AB']

    elif relation == 'BA':
        relation = ['BA']

    sampled_files = []
    sampled_files_name = []
    y_of_files = []

    for relation_ in relation:

        folder_path = os.path.join(base_dir, f'{relation_[0]}→{relation_[1]}')

        temp_file_path = os.path.join(base_dir, 'temp_file')
        if os.path.exists(temp_file_path):
            pass
        else:
            os.mkdir(temp_file_path)

        # 获取当前目录下所有以 'train' 开头并以 '.txt' 结尾的文件
        files = [f for f in os.listdir(folder_path) if f.startswith('train') and f.endswith('.txt')]
        # 定义用于记录处于阈值内文件的文件名
        record_file = rf'{base_dir}/temp_file/{relation_}_files_OUT_of_{threshold[0]}-{threshold[1]}.txt'
        print("Reading..")

        # 读取当前阈值的文件记录（如果存在）
        if os.path.exists(record_file):
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
            right_test_SAMPLE = len(valid_files)
            print('检测到: 符合阈值的文件数量 < 需要测试的数量')

        else:
            right_test_SAMPLE = test_SAMPLE

        # 随机选择50个文件（如果文件数小于50，选择所有文件）

        selected_files = random.sample(valid_files, min(test_SAMPLE//len(relation), right_test_SAMPLE))

        read_in_files = [pd.read_csv(os.path.join(folder_path, file), sep='\s+', header=None, dtype=float) for file in
                         selected_files]
        file_names = [relation_ + "_" + name_[:-4] for name_ in selected_files]

        sampled_files.extend(read_in_files)
        sampled_files_name.extend(file_names)

        if relation_ == 'AB':
            y_of_files.extend([1]*len(read_in_files))
        elif relation_ == 'BA':
            y_of_files.extend([0]*len(read_in_files))
        else:
            raise NotImplementedError

    return sampled_files, sampled_files_name, y_of_files, X, y


if __name__ == '__main__':

    read_files = return_cause_pair(relation='AB&BA', threshold=[0,500], test_SAMPLE=50, seed=5)
    a=1