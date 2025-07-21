import itertools
from typing import Literal, Union, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from tqdm import tqdm

from cause_pair_functions import test_tools_v312
from cause_pair_functions.Pow_method_PLSR import PLS_R
from cause_pair_functions.test_tools_v312 import spliter, distance_matrix

Distance_Option = Literal['Mah', 'Euc', 'Pairs', 'Pairs_T', 'Ming', 'origin']
Distance_Option_Iterable = Union[Distance_Option, Iterable[Distance_Option]]


# 求距离矩阵函数
def DPLS_distance(X, distance_pattern: Distance_Option_Iterable, dtype='float', **kwargs) -> np.ndarray:
    """

    :param X: 自变量

    :param distance_pattern: 距离矩阵模式

        假设 file_values 形状 : 100*5

        | 'Euc': 欧氏距离, 形状 100*100
        | 'Mah': 曼哈顿距离, 形状 100*500
        | 'Pairs': 两两距离的组合, 形状 100*1500
        | 'Ming': 明氏距离, 形状 100*100

        distance_pattern 输入形式要求为 list['mode1', 'mode2', ...]

        举例: 当输入 distance_pattern = ['tensor'], 则生成曼哈顿矩阵
             当输入 distance_pattern = ['tensor', 'vector'], 则生成 曼哈顿矩阵和欧式距离矩阵 按顺序拼接而成的矩阵


    :param dtype: complex:求复平面距离, float:求实数轴距离

    :param kwargs: 其他参数

    :return: distance 距离矩阵

    """

    if dtype is complex:

        angle_cosine = np.zeros((X.shape[0], X.shape[0]))

        for f in range(X.shape[0]):
            angle_cosine[:, f] = test_tools_v312.calculate_corr(X=X.T, y=X[f, :].T, R2=False)

        angle = np.arccos(angle_cosine)
        angle = np.nan_to_num(angle, nan=0)
        rotation = np.exp(1j * angle)

    else:

        rotation = 1

    # 根据 file_values, 按照 distance_pattern 构建 distance

    X_distance = []

    for distance_mode in distance_pattern:

        if distance_mode == 'Euc':

            X_distance_d = distance_matrix([X], distance_mode='Euc', dtype=dtype, **kwargs)

        elif distance_mode == 'Mah':

            X_distance_d = distance_matrix([X], distance_mode='Mah', dtype=dtype, **kwargs)

        elif distance_mode == 'Pairs':


            if X.shape[1] > 1:

                combinations = list(itertools.combinations(range(X.shape[1]), 2))

                pos_lst = [list(c) for c in combinations]
                x_list = [X[:, pos_i] for pos_i in pos_lst]

                X_distance_d = distance_matrix(x_list, distance_mode='Euc', dtype=dtype, **kwargs)

            else:

                X_distance_d = distance_matrix([X], distance_mode='Euc', dtype=dtype, **kwargs)

        elif distance_mode == 'Ming':

            X_distance_d = distance_matrix([X], distance_mode='Euc', dtype=dtype, distance_r=X.shape[1], **kwargs)

        elif distance_mode == 'origin':

            X_distance_d = [X]

        else:
            raise AttributeError("Do not know how to handle distance mode {}".format(distance_mode))

        X_distance_d = np.hstack(X_distance_d)
        X_distance.append(np.abs((X_distance_d * rotation).real))  # 尚未解决 Mah 情况

    X_distance_ary = np.hstack(X_distance)

    return X_distance_ary


# 求多折的距离矩阵函数
def DPLS_distance_divider(X: np.ndarray, cv: int, distance_pattern: Distance_Option_Iterable, **kwargs) -> [np.ndarray,
                                                                                                            list, list]:
    """

    :param X: 自变量
    :param cv: 独立测试折数
    :param distance_pattern: 距离矩阵模式
    :param kwargs: 其它参数

    :return: distance 距离矩阵, train_list 训练集索引, test_list 测试集索引

    """

    # 拆分 file_values 的训练集与测试集
    train_list, test_list = spliter(X.shape[0], cv=cv, **kwargs)

    # 定义 distance 的 train_list 和 test_list
    distance_train_list = []
    distance_test_list = []

    for train_index, test_index in zip(train_list, test_list):

        train_index = np.array(train_index)
        test_index = np.array(test_index)

        start_col = 0
        distance_train_list_d = []
        distance_test_list_d = []

        for distance_mode in distance_pattern:

            if distance_mode in ['Euc', 'Ming', 'cosine', 'Pairs_T']:

                train_index_d = train_index + start_col
                test_index_d = test_index + start_col

                start_col += X.shape[0]

            elif distance_mode == 'Mah':

                train_index_d = []
                test_index_d = []

                for train_i in train_index:
                    train_index_d.extend(list(range(train_i * X.shape[1], (train_i + 1) * X.shape[1])))

                for test_i in test_index:
                    test_index_d.extend(list(range(test_i * X.shape[1], (test_i + 1) * X.shape[1])))

                train_index_d = np.array(train_index_d)
                test_index_d = np.array(test_index_d)

                train_index_d += start_col
                test_index_d += start_col

                start_col += X.shape[0] * X.shape[1]

            elif distance_mode == 'Pairs':

                train_index_d = []
                test_index_d = []

                pairs_num = len(list(itertools.combinations(range(X.shape[1]), 2)))

                for i in range(pairs_num):
                    train_index_d.extend(train_index + i * X.shape[0] + start_col)
                    test_index_d.extend(test_index + i * X.shape[0] + start_col)

                start_col += pairs_num * X.shape[0]

            elif distance_mode == 'origin':

                train_index_d = list(range(start_col, X.shape[1] + start_col))
                test_index_d = list(range(start_col, X.shape[1] + start_col))

                start_col += X.shape[1]

            else:
                raise NotImplementedError

            distance_train_list_d.extend(train_index_d)
            distance_test_list_d.extend(test_index_d)

        distance_train_list.append(distance_train_list_d)
        distance_test_list.append(distance_test_list_d)

    return distance_train_list, distance_test_list


# PLS系数求和器
def sum_P(P, converge_iter, max_iter, y_dim):
    """

    :param P: PLS系数向量矩阵
    :param converge_iter: PLS 收敛的迭代阶数
    :param max_iter: 人为限制的最大迭代阶数, 如果迭代 max_iter 次还未收敛,则结束 PLS 过程
    :param y_dim: y 的数量, 通常为 1

    :return: PLS_p 系数向量矩阵

    """

    if P.shape[1] % y_dim != 0:
        raise 'Number of columns is not a multiple of y_dim'

    P_split = np.split(P, P.shape[1] // y_dim, axis=1)

    # coef_list 第 n 项是 P 的前 n 列之和, n <= converge_iter

    coef_list = []
    for i in range(converge_iter):
        i += 1
        coef_i = np.sum(P_split[:i], axis=0)
        coef_list.append(coef_i)

    # converge_iter 之后的列直接复制 coef_list 最后一项即可

    if converge_iter < max_iter:
        tile = np.tile(coef_list[-1], max_iter - converge_iter)
        coef_list.append(tile)

    return np.hstack(coef_list)


def DPLS_PLS_core(X, y, cv, distance_pattern: Distance_Option_Iterable, max_iter, tolerance, **kwargs):
    """
    :param X: 自变量
    :param y: 因变量
    :param cv: 独立测试折数
    :param distance_pattern: 距离矩阵模式
    :param max_iter: 最大迭代阶数
    :param tolerance: PLS 收敛容忍度, 迭代过程中两代之间差距 < tolerance 则程序判定 PLS 过程 已收敛
    :param kwargs: 其它参数

    :return: DPLS_R2

    """

    # file_values 训练集测试集索引
    train_list, test_list = spliter(X.shape[0], cv=cv, **kwargs)

    # distance 训练集测试集索引
    distance = DPLS_distance(X, distance_pattern=distance_pattern, **kwargs)
    distance_train_list, distance_test_list = DPLS_distance_divider(X, cv, distance_pattern=distance_pattern, **kwargs)

    # 定义 y_preds, 用于存储每个 p 对应的y_pred
    y_preds = np.tile(np.zeros_like(y), (1, max_iter))

    for i in range(cv):
        X_train = distance[train_list[i], :][:, distance_train_list[i]]
        X_test = distance[test_list[i], :][:, distance_train_list[i]]
        # 想想为什么是 distance_train_list[i], 而不是(这里↑ ) distance_test_list[i]

        y_train = y[train_list[i], :]

        # 距离矩阵使用PCA变为方形

        distance_PCA = PCA(n_components=X_train.shape[0]).fit(X_train)
        distance_PCA_coef = distance_PCA.components_

        distance_train = X_train @ distance_PCA_coef.T
        distance_test = X_test @ distance_PCA_coef.T

        # 求 P,e,r 分别为 迭代系数矩阵, 残差, PLS收敛的迭代层数
        P, e, r = PLS_R(distance_train, y_train, max_iter=max_iter, tolerance=tolerance)

        # p_ary: 系数矩阵
        p_ary = sum_P(P, r, max_iter, y_train.shape[1])

        # 计算第i折的 y_pred
        y_preds[test_list[i], :] = distance_test @ p_ary

    if y.shape[1] == 1:

        # 求 y_preds 每一列的 R2
        y_pred_R2 = test_tools_v312.calculate_corr(y_preds, y)

    else:
        raise ValueError('More than one y has been detected, 前面的领域以后再来探索吧')

    # 把 Nan 值用 0 填充
    y_pred_R2 = np.nan_to_num(y_pred_R2, nan=0)

    # R2 最大值即为 DPLS_R2
    max_pos = np.argmax(y_pred_R2)
    DPLS_R2 = float(y_pred_R2[max_pos])

    return DPLS_R2


def DPLS_direct(X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame], cv: int,
                *,
                distance_pattern: Union[list, str] = 'tensor',
                max_iter: int = 20,
                tolerance: float = 0.01,
                n_jobs: int = 1,
                bar: bool = True,
                DPLS_core: Literal['PLS', 'PCA_PLS'] = 'PLS',
                R_mode: Literal['fusion', 'single'] = 'single',

                **kwargs) -> [float]:
    """

    :param X: 自变量
    :param y: 因变量
    :param cv: 独立测试折数
    :param distance_pattern: 距离矩阵模式
    :param max_iter: 最大迭代阶数
    :param tolerance: PLS 收敛容忍度, 迭代过程中两代之间差距 < tolerance 则程序判定 PLS 过程 已收敛
    :param n_jobs: 多线程 线程数
    :param bar: 是否显示 进度条
    :param DPLS_core:  -
    :param R_mode: 返回 R 的模式, fusion: 将 file_values 视为一个整体, 返回(这个整体的) 1 个 R, single: 按列处理 file_values, 每列返回 1 个R
    :param kwargs: 其他参数
    :return: R_mode == fusion: [DPLS_R2], R_mode == single: [DPLS_R2_1, ...,DPLS_R2_n], n is file_values.shape[1]

    """

    if isinstance(distance_pattern, str):
        distance_pattern = [distance_pattern]

    # 化为 2D_ary
    X = test_tools_v312.to_2D_ary(X)
    y = test_tools_v312.to_2D_ary(y)

    if DPLS_core == 'PLS':

        DPLS_func = DPLS_PLS_core

    else:
        raise NotImplementedError

    if R_mode == 'fusion':

        DPLS_R = DPLS_func(X=X, y=y, cv=cv, distance_pattern=distance_pattern, max_iter=max_iter, tolerance=tolerance,
                           **kwargs)

        return DPLS_R

    elif R_mode == 'single':

        R_list = Parallel(n_jobs=n_jobs)(delayed(DPLS_func)(X=X[:, i].reshape(-1, 1), y=y, cv=cv,
                                                            distance_pattern=distance_pattern,
                                                            max_iter=max_iter,
                                                            tolerance=tolerance, **kwargs) for i in
                                         tqdm(range(X.shape[1]),
                                              desc='-DPLS_direct',
                                              total=X.shape[1],
                                              ncols=86,
                                              colour='white',
                                              disable=not bar))

        return R_list

    else:
        raise ValueError('R_mode must be either "fusion" or "single"')


# if __name__ == '__main__':
#
#     from sklearn.cross_decomposition import PLSRegression
#     from Projects.Muti_func_creat.muti_func_test import add_noise
#
#     #
#     # sample_num=1000
#     # features_num=2
#     #
#     #
#     # def do_PLS (f, s):
#     #
#     #
#     #     np.random.seed(None)
#     #     X_i = np.random.rand(s, f).astype(float)
#     #     y_i = np.random.rand(s, 1).astype(float)
#     #     y_real = np.prod(X_i, axis=1)
#     #     X_d = distance_matrix([X_i], distance_mode='Euc', distance_r=f)[0]
#     #
#     #     # y_create_obs = add_noise(y_i, 0.5)
#     #     # print(test_tools_v312.calculate_corr(y_i, y_create_obs))
#     #
#     #     result_i = PLSRegression(n_components=500).fit_transform(X_d, y_i)[0]
#     #     result_real = PLSRegression(n_components=500).fit_transform(X_d, y_real)[0]
#     #     r_i = test_tools_v312.calculate_corr(result_i, y_i)
#     #     r_real = test_tools_v312.calculate_corr(result_real, y_i)
#     #
#     #     a = sum_P(np.array(r_i).reshape(1, -1), 500, 500, 1)
#     #     b = sum_P(np.array(r_real).reshape(1, -1), 500, 500, 1)
#     #
#     #     plt.scatter(np.arange(1, 501), a, label='0_hypothesis', color='blue', marker='o',)
#     #     plt.scatter(np.arange(1, 501), b, label='y_exp', color='red', marker='o',)
#     #     plt.legend()
#     #     plt.show()
#     #
#     #     return
#     #
#     # do_PLS(5, 500)
#
#     # for f_ in [2]:
#     #
#     #     result_list = Parallel(n_jobs=8)(delayed(do_PLS)(f=f_, s=s_) for s_ in tqdm([500]*10000))
#     #     result = pd.DataFrame(np.vstack(result_list))
#     #     result.to_excel(f'500sample_PLS_零假设_{f_}x_250p_10000times.xlsx')
#
#     # print(np.vstack(para_result))
#
#     # R_ = DPLS_direct(file_values=X_, y=y_, cv=5, DPLS_core='PLS', R_mode='fusion', distance_pattern=['vector', 'pairs', 'tensor'], n_jobs=1)
#     # print(R_)
#
#     sample_num = 500
#     feature_num = 4
#     file_values = np.random.rand(sample_num, feature_num)
#     D = test_tools_v312.distance_matrix([file_values], distance_mode='Euc', distance_r=file_values.shape[1])[0]
#     y0 = np.prod(file_values, axis=1).reshape(-1, 1)
#     y0 = add_noise(y0, 0)
#     y1 = np.random.rand(sample_num, 1)
#     all_sample_feature_res = np.zeros((2, 100))
#
#     P = np.arange(1, 100 + 1)
#     m1 = [2.21169467, -2.3817057, 0.984505]
#     m2 = [3.02072028, -2.9587557, 0.981375]
#     m3 = [2.99084683, -3.1657871, 0.981389]
#     m4 = [2.44274849, -3.164627, 0.985082]
#     m5 = [2.227481, -3.2405129, 0.980818]
#
#     m = [m1, m2, m3, m4, m5]  # lame!
#     Fit_R2P = 1 / (1 + np.exp(-1 * (m[file_values.shape[1] - 1][0]) * (np.log(P / file_values.shape[0]) - (m[file_values.shape[1] - 1][1]))))
#
#     pls_t1 = PLSRegression(n_components=100).fit_transform(D, y0)[0]
#     y_obj_1 = PLSRegression(n_components=100, ).fit(D, y0)
#
#     pls_q1 = y_obj_1.y_loadings_
#     pls_w1 = y_obj_1.x_rotations_
#
#     p_ary = pls_w1 * pls_q1
#     y_pred_ = test_tools_v312.stdize(D) @ sum_P(p_ary, 100, 100, 1)
#     # y_pred = y_obj_1.predict(D)
#     r_i_ = test_tools_v312.calculate_corr(y_pred_, y0)
#     # a = sum_P(np.array(r_i_).reshape(1, -1), 100, 100, 1)
#     all_sample_feature_res[0, :] = r_i_
#
#     #
#     for i in range(100):
#         pls_obj = PLSRegression(n_components=i + 1).fit(D, y0)
#         y_pred = pls_obj.predict(D)
#         a = test_tools_v312.calculate_corr(y_pred, y0)[0]
#
#         all_sample_feature_res[1, i] = a
#
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
#     plt.rcParams['axes.unicode_minus'] = False  # 显示负号
#     plt.scatter(np.arange(100), all_sample_feature_res[0, :], label="pred", alpha=0.5)
#     plt.scatter(np.arange(100), all_sample_feature_res[1, :], label="零假设_t", alpha=0.5)
#     plt.legend()
#     plt.show()
