import random
import traceback
from typing import Literal
from typing import Tuple

import numpy as np
import pandas as pd


def gen_x(sample_num:int=500, param_num:int=2, x_start=-1, x_end=1, x_mode:Literal['uniform', 'grow', 'parabola']='uniform', x_seed=None, **kwargs) -> pd.DataFrame:

    x_i_list = []
    gen_x_seed = gen_seed(param_num=param_num, rand_seed=x_seed)[0]

    for feature in range(param_num):

        np.random.seed(gen_x_seed[feature])
        # 👇随机种子
        if x_mode == 'uniform':
            x_i = np.around(np.random.uniform(x_start, x_end, size=sample_num), 8)

        elif x_mode == 'grow':
            x_grow = np.around(np.random.uniform(0, (x_end + (0 - x_start)) ** 3, size=sample_num), 8)
            x_i = x_grow ** (1 / 3)
            x_i += x_start

        elif x_mode == 'parabola':
            x_i = np.around(np.random.uniform(x_start, x_end, size=sample_num), 8)
            x_i = x_i * (np.abs(x_i) / np.abs(x_end - x_start))

        else:
            raise AttributeError(f"无法识别x_mode{x_mode}")

        x_i_list.append(x_i.reshape(-1, 1))

    # 释放种子
    np.random.seed(None)

    x = np.hstack(x_i_list)
    x = pd.DataFrame(x, columns=[f'x_{i+1}' for i in range(param_num)])
    
    return x


def gen_redun_feature(x, func_seed, noise, redun_x_num, redun_x_to_x_num, x_to_x_level = 0):

    X_redun, redun_picked = cal_X(x=x, use_x_num=redun_x_num, x_to_x_num=redun_x_to_x_num, func_seed=func_seed, x_to_x_level=x_to_x_level)

    # 构建方程名
    func_name = 'y=' + '+'.join(list(X_redun.columns))
    x_redun_obs = add_noise(np.sum(X_redun.to_numpy(), axis=1), noise_degree=noise, func_seed=func_seed)

    x_redun = pd.DataFrame(x_redun_obs, columns=[func_name])

    return x_redun


def gen_redun_x(x: pd.DataFrame, redun_seed=0, redun_ratio = 1,  x_to_x_level = 3, x_sampled=None, redun_max_count=5, **kwargs):


    np.random.seed(redun_seed)
    random.seed(redun_seed)

    if x_sampled is not None:
        pass
    else:
        x_sampled = x.columns.tolist()

    x_contain_redun = x[x_sampled]
    x_redun_slope = generate_slop(x_contain_redun.shape[1], slop_2=redun_ratio)
    redun_count = random.choice(x_redun_slope)
    # # print('x_shape', x_contain_redun.shape)
    # print("redun_count", redun_count)
    # print("redun_slope", x_redun_slope)
    count=0

    try:

        while redun_count > 0:

            if count > redun_max_count:
                break

            redun_seeds = gen_seed(redun_count, rand_seed=(redun_seed+count))[0]

            count += 1
            # print("redun_seeds", redun_seeds)
            redun_ratio = (redun_count + x_contain_redun.shape[1]) / (x_contain_redun.shape[1] + count)

            x_contain_redun_iter = x_contain_redun

            for seed_i in redun_seeds:

                np.random.seed(seed_i)
                random.seed(seed_i)

                x_num = random.choice(generate_slop(x.shape[1]-1, redun_ratio, translation=1), )
                x_to_x_num = random.choice(generate_slop(x.shape[1], redun_ratio))

                x_redun_i, redun_func = cal_X(x_contain_redun,
                                              func_seed=seed_i,
                                              use_x_num=x_num,
                                              x_to_x_num=x_to_x_num,
                                              x_to_x_level=x_to_x_level, **kwargs)
                # x_redun_i_obs = add_noise(x_redun_i, noise_degree=0.45, func_seed=seed_i)

                # r2_scores = [r2_score(x_redun_i.to_numpy()[:, j], x_redun_i_obs.to_numpy()[:, j]) for j in range(x_redun_i.shape[1])]
                # print('redun_R2: ', r2_scores)

                x_contain_redun_iter = pd.concat([x_contain_redun_iter, x_redun_i], axis=1)
                x_contain_redun_iter = x_contain_redun_iter.T.drop_duplicates().T

            x_contain_redun = x_contain_redun_iter

            # if count==2:
            #     print("-------------------------------------------")
            #     print("x_contain_redun_col", x_contain_redun.columns.tolist())

            random.seed(redun_seed+count)
            redun_count = random.choice(generate_slop(x_contain_redun.shape[1], redun_ratio), )

            # print("redun_ratio", redun_ratio)
            # print("redun_count", redun_count)

    except Exception as e:
        print(e)
        traceback.print_exc()
    return x_contain_redun



def gen_seed(param_num: int, rand_seed: int, gen_times: int=1):
    np.random.seed(rand_seed)
    seed_list = np.random.randint(0, 50000, [gen_times, param_num]).reshape(gen_times, param_num)
    return seed_list.tolist()


def generate_slop(n:int, slop_2=2.5, translation=0):

    if n < 0:
        print('generate_list param: n has been adjusted to 0')
        n = 0

    if slop_2 > 8:
        print('generate_list param: slop_1 has been limited to 8')
        slop_2 = 8

    x_num_list = []

    if n == 0:

        x_num_list = [0]

    else:
        coef = slop_2
        coef_ = coef / (n ** (1/coef))

        for k in range(0, n + 1):  # 从 2 到 n

            d = (n+1) - k

            x_num_list.extend([k] * int(d ** ((coef - (k ** (1/coef))*coef_) / (coef**(np.log(coef)/ np.log(np.pi))))))  # 每个 k 重复 k-1 次

    return np.array(x_num_list) + translation


def x_function(x: pd.DataFrame, use_x_num:int, x_num:int, func_seed:int, linear:bool=False, use_x_func = None,
               linear_coef_range:Tuple[float, float] = (-3, 3), linear_intercept_range: Tuple[float, float] = (-5, 5), x_sampled=None, **kwargs):
    
    X = pd.DataFrame()
    random.seed(func_seed)

    linear_coef = random.uniform(*linear_coef_range)
    linear_intercept = random.uniform(*linear_intercept_range)

    function_dict = {
        "线性函数":lambda f: linear_coef * f + linear_intercept,
        "正弦函数": lambda f: np.sin(np.pi*f),
        "余弦函数": lambda f: np.cos(np.pi * f),
        "二次函数": lambda f: 2 * (f ** 2),
        "平方根函数": lambda f: np.sqrt(np.abs(f)),
        "指数函数": lambda f: np.exp(f),
        "对数函数（平移）": lambda f: np.log(f + 1),
        "对数函数（加偏移防负值）": lambda f: np.log(np.abs(f) + 1e-10),
        "Sigmoid 函数": lambda f: 1 / (1 + np.exp(-6 * f)),
        "三次多项式函数": lambda f: 2 * (f ** 3) + f ** 2 - 2 * f,
        "指数幂函数": lambda f: 2 ** (5 * (f + 1)),
        "高频正弦函数": lambda f: np.sin(6 * np.pi * f),
        "混合三角+线性函数": lambda f: 0.2 * np.sin(4 * f) + (11 / 10) * f,
        "高频正弦 + 线性项": lambda f: np.sin(5 * np.pi * f) + f,
        "高频余弦函数": lambda f: np.cos(6 * np.pi * f),
        "高频正弦线性混合函数": lambda f: (1 / 10) * np.sin(10.6 * f) + (11 / 10) * f,
        "非线性频率余弦函数": lambda f: np.cos(6 * np.pi * f * (f + 1)),
        "非线性频率正弦函数": lambda f: np.sin(6 * np.pi * f * (f + 1)),
    }

    if use_x_func is None:

        use_x_func = ["正弦函数", "余弦函数", "二次函数", "平方根函数"]

    if linear:

        use_x_func = ["线性函数"]


    if x_sampled is not None:
        pass

    else: x_sampled = x.columns.tolist()

    def select_func():
        # 定义函数列表及其对应的字符串表示


        functions_formula = {
            "线性函数": f'(({linear_coef:.2f}){x_mark}＋({linear_intercept:.2f}))',
            "正弦函数": f'sin(π{x_mark})',
            "余弦函数": f'cos(π{x_mark})',
            "二次函数": f'2({x_mark})^2',
            "平方根函数": f'sqrt({x_mark})',
            "指数函数": f'e^({x_mark})',
            "对数函数（平移）": f'log({x_mark}＋1)',
            "对数函数（加偏移防负值）": f'log({x_mark})',
            "Sigmoid 函数": f'1/(1＋e^-6{x_mark})',
            "三次多项式函数": f'2{x_mark}^3＋{x_mark}^(2)-2{x_mark}',
            "指数幂函数": f'2^5({x_mark}＋1)',
            "高频正弦函数": f'sin(2π{x_mark})',
            "混合三角+线性函数": f'1/5×sin(4x)＋(11/10)×{x_mark}',
            "高频正弦 + 线性项": f'sin(5π{x_mark})＋{x_mark}',
            "高频余弦函数": f'cos(6π{x_mark})',
            "高频正弦线性混合函数": f'1/10×sin(10.6×{x_mark})＋(11/10)×{x_mark}',
            "非线性频率余弦函数": f'cos(5πx({x_mark}＋1))',
            "非线性频率正弦函数": f'sin(4πx({x_mark}＋1))'
        }

        functions = []
        for use_func in use_x_func:

            functions.append((function_dict[use_func], functions_formula[use_func]))

        return random.choice(functions)

    funcs_seed = gen_seed(param_num=x_num, rand_seed=func_seed)[0]  # seed -> [seed_1, seed_2, ..., seed_i], i=use_x_num


    if use_x_num < len(x_sampled):
        pass
    else:
        use_x_num = len(x_sampled)


    if use_x_num >= x_num:
        x_sampled = random.sample(x_sampled, x_num)

    else:
        x_sampled_ = random.sample(x_sampled, k=use_x_num)
        x_sampled = random.choices(x_sampled_, k=x_num-use_x_num)
        x_sampled.extend(x_sampled_)

    # print("x_sampled =", x_sampled, "use_x_num =", use_x_num, 'x_num =', x_num)

    # 随机选择一个函数及其字符串表示
    for x_mark, seed_ in zip(x_sampled, funcs_seed):

        np.random.seed(seed_)
        func, func_name = select_func()

        X_i = func(x[x_mark].to_numpy())
        X = pd.concat([X, pd.DataFrame(X_i, columns=[func_name])], axis=1)

    # 释放种子
    random.seed(None)
    np.random.seed(None)

    # 返回函数计算结果和函数名称
    return X, list(set(x_sampled))


def x_to_x_function(x: pd.DataFrame, func_seed, x_to_x_num, x_sampled=None, x_to_x_level=3,use_xtox_func = None, **kwargs):
    
    X = pd.DataFrame()

    xtox_func_dict = {
    "和函数": lambda f: np.sum(f, axis=1),
    "绝对值和函数": lambda f: np.abs(np.sum(f, axis=1)),
    "正弦和函数": lambda f: np.sin(np.sum(f, axis=1)),
    "余弦和函数": lambda f: np.cos(np.sum(f, axis=1)),
    "正弦积函数": lambda f: np.sin(np.abs(np.prod(f, axis=1))),
    "余弦积函数": lambda f: np.cos(np.abs(np.prod(f, axis=1))),
    "积函数": lambda f: np.prod(f, axis=1),
    "指数积函数": lambda f: np.exp(np.prod(f, axis=1)),
    "除函数": lambda f: f[:, 0] / np.prod(f, axis=1),
    }

    if use_xtox_func is None:

        use_xtox_func = ["积函数"]


    def select_func(x_picked_id):


        xtox_formular_dict = {
            "和函数": f"({'＋'.join(x_picked_id)})",
            "绝对值和函数": f"abs(" + '＋'.join(x_picked_id) + ')',
            "正弦和函数": f"sin(" + '＋'.join(x_picked_id) + ')',
            "余弦和函数": f"cos(" + '＋'.join(x_picked_id) + ')',
            "正弦积函数": f"sin(" + '×'.join(x_picked_id) + ')',
            "余弦积函数": f"cos(" + '×'.join(x_picked_id) + ')',
            "积函数": '(' + '×'.join(x_picked_id) + ')',
            "指数积函数": 'e^(' + '×'.join(x_picked_id) + ')',
            "除函数": '(' + '/'.join(x_picked_id) + ')',
        }

        xtox_functions = []
        for use_func in use_xtox_func:
            xtox_functions.append((xtox_func_dict[use_func], xtox_formular_dict[use_func]))

        return random.choice(xtox_functions)

    funcs_seed = gen_seed(param_num=x_to_x_num, rand_seed=func_seed)[0]
    x_num_list = generate_slop(x_to_x_level-2, translation=2)

    x_pickeds = []

    if x_sampled is None:
        x_sampled = x.columns.tolist()

    else:
        pass

    # print("x_sampled in xtox =", x_sampled)

    for seed_f in funcs_seed:

        np.random.seed(seed_f)
        random.seed(seed_f)
        
        # 从 x_num_list 中 选择 use_x_num
        x_num_in_one = random.choice(x_num_list)

        if len(x_sampled) >= x_num_in_one:

            x_picked_ID= random.sample(x_sampled, x_num_in_one)

            # print("len(x_sampled) >= x_num_in_one, x_picked_ID=", x_picked_ID)

        else:

            x_picked_ID = random.choices(x_sampled, k=x_num_in_one - len(x_sampled))
            x_picked_ID.extend(x_sampled)

            # print("len(x_sampled) < x_num_in_one, x_picked_ID=", x_picked_ID)

        x_picked = x[x_picked_ID].to_numpy()
        x_pickeds.extend(x_picked_ID)

        func, func_name = select_func(x_picked_ID)

        X_i = func(x_picked)
        X = pd.concat([X, pd.DataFrame(X_i, columns=[func_name])], axis=1)

    random.seed(None)

    return X, list(set(x_pickeds))



def cal_X(x:pd.DataFrame, func_seed:int, use_x_num=5, x_num=5, x_to_x_num:int = 0, x_to_x_level = 3, x_sampled = None, **kwargs):

    X_single_DF = pd.DataFrame()
    X_multi_DF = pd.DataFrame()
    x_pickeds = []


    # -----------------------------------------------------------------------------------------------------------------

    if use_x_num > 0:

        X_single_DF, x_sampled = x_function(x, use_x_num=use_x_num, func_seed=func_seed,x_num=x_num,x_sampled=x_sampled, **kwargs)

        x_pickeds.extend(x_sampled)


    if x_to_x_num > 0:

        sample_gap = use_x_num - len(x_sampled)
        apply_sample = random.sample(list(set(x.columns.tolist()) - set(x_sampled)), sample_gap)

        if apply_sample:
            x_sampled = apply_sample
        else:
            pass

        X_multi_DF, x_to_x_sampled = x_to_x_function(x, func_seed, x_to_x_num=x_to_x_num, x_to_x_level=x_to_x_level, x_sampled=x_sampled, **kwargs)
        x_pickeds.extend(x_to_x_sampled)



    return pd.concat([X_single_DF, X_multi_DF], axis=1), list(set(x_pickeds))



def add_noise(_exp: np.ndarray | pd.DataFrame, noise_degree:float=0.5, func_seed=1,  **kwargs):

    np.random.seed(func_seed)

    if noise_degree == 0 or noise_degree == 0.0:
        return _exp
    noise_degree = np.abs(noise_degree)

    _obs = _exp + np.random.normal(size=_exp.shape, loc=0, scale=_exp.std() * noise_degree)

    np.random.seed(None)
    return _obs


def gen_y_exp(sample_num:int=500, param_num:int=2,
              use_x_num:int=2, x_to_x_num:int=0, x_num=2,
              redundancy:bool=False,
              func_seed=None, x_seed=None, **kwargs):

    """

    :param sample_num: 样本数
    :param param_num: 特征数
    :param use_x_num: 参与函数构成的项数
    :param redundancy: 是否产生冗余特征, 当 x2 = f(x1) + ε 称 x2 冗余

    :param x_seed: 随机种子, 用于生成 x
    :param func_seed: 随机种子, 决定总体的关系 f

    :param x_to_x_num: 互作项数

    :param kwargs:

    :return: x: 所有特征, x_picked: 参与 _exp 构成的 x, X: f(x), _exp: 因变量
    """

    if use_x_num > param_num:
        raise AttributeError('use_x_num must <= param_num')

    x = gen_x(sample_num=sample_num, param_num=param_num,x_seed=x_seed, **kwargs)

    if redundancy:

        x_contain_redun = gen_redun_x(x, **kwargs)

        X_contain_redun, x_picked_redun = cal_X(x_contain_redun, func_seed=func_seed, use_x_num=use_x_num, x_to_x_num=x_to_x_num, x_num=x_num,  **kwargs)

        y_exp = np.sum(X_contain_redun.to_numpy(), axis=1)

        return x_contain_redun, X_contain_redun, y_exp, x_picked_redun

    else:

        X, x_picked = cal_X(x, func_seed=func_seed, use_x_num=use_x_num, x_to_x_num=x_to_x_num, x_num=x_num, **kwargs)

    y_exp = np.sum(X.to_numpy(), axis=1)

    return x, X, y_exp, x_picked



if __name__ == '__main__':

    s=800
    f_=6
    times_ = 10000

    for i in range(10):

        x_, X_, y_exp_, x_picked_ = gen_y_exp(500, 10, 4, 1, x_num=1, x_to_x_level=3, redundancy=True, func_seed=1, x_seed=1, redun_seed=3, redun_ratio=6)
        # print(i)
        # print(X_.columns)














