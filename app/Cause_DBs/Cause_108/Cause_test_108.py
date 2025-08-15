import os

from GUI_functions.casual_pair_tester import return_Cause_DF, Pre_process_Iterable, Method_Option
from pair_data_presenter import return_cause_pair
from typing import Literal
import random
import numpy as np
import pandas as pd


Relation: Literal['AB', 'BA'] = 'AB&BA'

test_seed = 7
random.seed(test_seed)  # 设置随机种子
np.random.seed(test_seed)
Thread = 1  # 多线程不能调试改成 1-

Threshold = [400, 500]
Test_flies_num = 100

Kwargs = {'need_P': True, 'need_Rs': True}
Pre_process: Pre_process_Iterable = []

if isinstance(Pre_process, list):
    Pre_process_name = '-'.join(Pre_process)
else:
    Pre_process_name = Pre_process

if Kwargs == {}:
    Kwargs_name: str = ''
else:
    Kwargs_name:list = [f"{key}={value}" for key, value in Kwargs.items()]
    Kwargs_name = '_{' + '-'.join(Kwargs_name) + '}'

Method: Method_Option = 'DPLSR'

read_files, file_names, file_causes = return_cause_pair(relation=Relation, threshold=Threshold, test_SAMPLE=Test_flies_num, seed=test_seed)

result_list = []

values_return = return_Cause_DF(file_value_list=read_files, reverse=False, thread=Thread, pre_process=Pre_process, seed=test_seed,
                              method=Method, **Kwargs)
values_Re_return = return_Cause_DF(file_value_list=read_files, reverse=True, thread=Thread, pre_process=Pre_process, seed=test_seed,
                            method=Method, **Kwargs)

result_DF = pd.concat([values_return, values_Re_return], axis=1)
result_DF.index = file_names

if os.path.exists(rf'analysis/{Method}'):
    pass
else:
    os.mkdir(rf'analysis/{Method}')
    print(f'已创建临时文件夹 analysis/{Method}')
print(result_DF)
result_DF.to_excel(rf'analysis/{Method}/{Relation}_{Threshold[0]}-{Threshold[1]}_{Test_flies_num}-Cause-108_{Pre_process_name}{Kwargs_name}_{Method}.xlsx')