import os

from functions.Casual_Pair_tester import return_Cause_DF, Pre_process_Iterable, Method_Option
from typing import Literal
import random
import numpy as np
import pandas as pd
from pair_data_presenter import return_cause_pair


Relation: Literal['AB', 'BA'] = 'AB'

test_seed = 7
random.seed(test_seed)  # 设置随机种子
np.random.seed(test_seed)
Thread = 8  # 多线程不能调试改成 1-

Kwargs = {'Sort_by': 'reason'}
Pre_process: Pre_process_Iterable = 'normalize'

if isinstance(Pre_process, list):
    Pre_process_name = '-'.join(Pre_process)
else:
    Pre_process_name = Pre_process

if Kwargs == {}:
    Kwargs_name: str = ''
else:
    Kwargs_name:list = [f"{key}={value}" for key, value in Kwargs.items()]
    Kwargs_name = '_{' + '-'.join(Kwargs_name) + '}'

Method: Method_Option = 'PATH'

suppercon_pair_format, col_names = return_cause_pair()

result_list = []

values_return = return_Cause_DF(file_value_list=suppercon_pair_format, reverse=False, thread=Thread, pre_process=Pre_process, seed=test_seed,
                              method=Method, **Kwargs)
values_Re_return = return_Cause_DF(file_value_list=suppercon_pair_format, reverse=True, thread=Thread, pre_process=Pre_process, seed=test_seed,
                            method=Method, **Kwargs)

result_DF = pd.concat([values_return, values_Re_return], axis=1)
result_DF.index = col_names

if os.path.exists(rf'analysis/{Method}'):
    pass
else:
    os.mkdir(rf'analysis/{Method}')
    print(f'已创建临时文件夹 analysis/{Method}')

result_DF.to_excel(rf'analysis/{Method}/{Relation}_Supercon_{Pre_process_name}{Kwargs_name}_{Method}.xlsx')