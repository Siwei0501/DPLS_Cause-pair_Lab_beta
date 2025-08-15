import os

from GUI_functions.casual_pair_tester import return_Cause_DF, Pre_process_Iterable, Method_Option
from typing import Literal
import random
import numpy as np
import pandas as pd


Relation: Literal['AB', 'BA'] = 'AB'

test_seed = 7
random.seed(test_seed)  # 设置随机种子
np.random.seed(test_seed)
Thread = 1  # 多线程不能调试改成 1-

Kwargs = {}
Pre_process: Pre_process_Iterable = 'regionalitze'

if isinstance(Pre_process, list):
    Pre_process_name = '-'.join(Pre_process)
else:
    Pre_process_name = Pre_process

if Kwargs == {}:
    Kwargs_name: str = ''
else:
    Kwargs_name:list = [f"{key}={value}" for key, value in Kwargs.items()]
    Kwargs_name = '_{' + '-'.join(Kwargs_name) + '}'

Method: Method_Option = 'chain_stability'

airfoil_file = pd.read_csv(f'airfoil_self_noise.dat', sep='\s+', header=None)
file_names = airfoil_file.columns
airfoil_pair = [pd.concat([pd.DataFrame(airfoil_file.iloc[:, i]), pd.DataFrame(airfoil_file.iloc[:, -1])], axis=1) for i in range(airfoil_file.shape[1]-1)]
airfoil_pair_format = []

for pair in airfoil_pair:

    pair.columns = [0,1]
    airfoil_pair_format.append(pair)

result_list = []

values_return = return_Cause_DF(file_value_list=airfoil_pair_format, reverse=False, thread=Thread, pre_process=Pre_process, seed=test_seed,
                              method=Method, **Kwargs)
values_Re_return = return_Cause_DF(file_value_list=airfoil_pair_format, reverse=True, thread=Thread, pre_process=Pre_process, seed=test_seed,
                            method=Method, **Kwargs)

result_DF = pd.concat([values_return, values_Re_return], axis=1)
result_DF.index = file_names

if os.path.exists(rf'analysis/{Method}'):
    pass
else:
    os.mkdir(rf'analysis/{Method}')
    print(f'已创建临时文件夹 analysis/{Method}')

result_DF.to_excel(rf'analysis/{Method}/{Relation}_Airfoil_{Pre_process_name}{Kwargs_name}_{Method}.xlsx')