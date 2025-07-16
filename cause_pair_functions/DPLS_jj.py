from typing import Literal, Union, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from tqdm import tqdm

from cause_pair_functions import test_tools_v312
from cause_pair_functions.DPLS_Direct import DPLS_distance, DPLS_distance_divider, sum_P
from cause_pair_functions.Pow_method_PLSR import PLS_R
from cause_pair_functions.test_tools_v312 import spliter

Distance_Option = Literal['Mah', 'Euc', 'Pairs', 'Ming', 'origin']
Distance_Option_Iterable = Union[Distance_Option, Iterable[Distance_Option]]


class DPLS:

    def __init__(self,
                 *,
                 cv: int = 1,
                 distance_pattern: Distance_Option_Iterable = 'Euc',
                 dtype=float,
                 copy=True,
                 whiten=False,
                 max_iter=20,
                 bar=True,
                 eig_solver: Literal['pow', 'sklearn'] = 'sklearn',
                 pow_protect_coef=10,
                 R_mode: Literal['single', 'fusion'] = 'fusion',
                 tol=0.001,
                 n_jobs=1,
                 square=True,
                 random_state=None,
                 **kwargs):

        self.cv = cv
        self.R_mode = R_mode

        self.dtype = dtype
        self.copy = copy
        self.whiten = whiten
        self.max_iter = max_iter

        self.bar = bar
        self.eig_solver = eig_solver
        self.pow_protect_coef = pow_protect_coef
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.square = square

        # 预测过程属性

        self.X = None
        self.y = None
        self.fit_mode=None

        self.R2 = None
        self.cv_R2 = None
        self.fit_R2 = None

        self.p = None
        self.p_arys = None
        self.cv_p = None
        self.fit_p = None

        self.y_pred = None
        self.y_preds = None
        self.y_pred_R2 = None

        self.coef: np.ndarray | None = None
        self.coef_transform = None

        if isinstance(distance_pattern, str):
            self.distance_pattern = [distance_pattern]
        else:
            self.distance_pattern = distance_pattern

    def __getattr__(self, name):

        self.__dict__[name] = None
        return None

    def _DPLS_PLS_core(self, X, y, cv):

        """
        :param X: 自变量
        :param y: 因变量
        """

        storager = {

            "R2": None,
            "p": None,
            "p_arys": None,
            "y_pred": None,
            "y_preds": None,
            "y_pred_R2": None,
            "coef": None,

        }

        # 防止最大迭代数超过距离矩阵列数
        if self.max_iter > X.shape[0]:
            self.max_iter = X.shape[0]

        # X_ 训练集测试集索引
        train_list, test_list = spliter(X.shape[0], cv=cv)

        # distance 训练集测试集索引
        distance = DPLS_distance(X, distance_pattern=self.distance_pattern, dtype=self.dtype)
        distance_trian_list, distance_test_list = DPLS_distance_divider(X, cv,
                                                                        distance_pattern=self.distance_pattern)

        # 定义 y_preds, 用于存储每个 p 对应的y_pred
        y_preds = np.tile(np.zeros_like(y), (1, self.max_iter))
        p_arys = []

        try:
            for i in range(cv):

                X_train = distance[train_list[i], :][:, distance_trian_list[i]]
                X_test = distance[test_list[i], :][:, distance_trian_list[i]]
                # 想想为什么是 distance_trian_list[i], 而不是(这里↑) distance_test_list[i]

                y_train = y[train_list[i], :]

                # 把距离矩阵训练集变为方形, 如果 X_train 满秩, 此步骤不损失信息

                if self.square:

                    distance_train = X_train @ X_train.T
                    distance_test = X_test @ X_train.T

                else:

                    distance_train = X_train
                    distance_test = X_test

                if self.eig_solver == 'pow':

                    # 求 P,e,r 分别为 迭代系数矩阵, 残差, PLS收敛的迭代层数
                    P, e, r = PLS_R(distance_train, y_train, max_iter=self.max_iter, tolerance=self.tol)

                    # p_ary: 系数矩阵
                    p_ary = sum_P(P, r, self.max_iter, y_train.shape[1])

                elif self.eig_solver == 'sklearn':

                    distance_pls_obj = PLSRegression(n_components=self.max_iter, scale=False).fit(distance_train, y_train)
                    P = distance_pls_obj.x_rotations_ * distance_pls_obj.y_loadings_
                    p_ary = sum_P(P, self.max_iter, self.max_iter, y_train.shape[1])

                else:
                    raise AttributeError('eig_solver 的值仅限于pow或sklearn')

                # 记录系数矩阵 p_ary
                p_arys.append(p_ary)

                # 计算第i折的 y_pred
                y_preds[test_list[i], :] = (distance_test - np.mean(distance_test, axis=0)) @ p_ary

        except ValueError:

            print('非法的矩阵,PLSR奇异值分解过程不符合要求')

            return storager

        else:

            if y.shape[1] == 1:

                # 求 y_preds 每一列的 R2
                cv_pred_R2 = test_tools_v312.calculate_corr(y_preds, y)

            else:
                raise ValueError('More than one y_ has been detected, 前面的领域以后再来探索吧')

            # 把 Nan 值用 0 填充
            cv_pred_R2 = np.nan_to_num(cv_pred_R2, nan=0)

            # R2 最大值即为 DPLS_R2
            p = np.argmax(cv_pred_R2)
            DPLS_R2 = cv_pred_R2[p]
            coef = [p_ary_[:, p].reshape(-1, 1) for p_ary_ in p_arys]

            storager['R2'] = DPLS_R2
            storager['p'] = p
            storager['p_arys'] = p_arys
            storager['coef'] = coef

            storager['y_pred'] = y_preds[:, p].reshape(-1, 1)
            storager['y_preds'] = y_preds
            storager['y_pred_R2'] = cv_pred_R2

        return storager


    def _fit_rectify(self, **kwargs):

        storager = {}

        if self.fit_mode == 'CV':

            cv_stored = self._DPLS_PLS_core(self.X, self.y, self.cv)
            storager.update(cv_stored.copy())

        if self.fit_mode == 'Fit_rectify':

            cv_stored = self._DPLS_PLS_core(self.X, self.y, self.cv)
            storager.update(cv_stored.copy())

            fit_stored = self._DPLS_PLS_core(self.X, self.y, cv=1)

            if fit_stored["R2"] is np.nan:

                return fit_stored

            fit_pred_R2 = fit_stored['y_pred_R2']
            fit_y_preds = fit_stored['y_preds']

            P = np.arange(1, self.max_iter + 1)

            # 好美的调试 ----------------------------------------------------------------------------------------------------------

            # average
            m1 = [2.21169467, -2.3817057, 0.984505]
            m2 = [3.02072028, -2.9587557, 0.981375]
            m3 = [2.99084683, -3.1657871, 0.981389]
            m4 = [2.44274849, -3.164627, 0.985082]
            m5 = [2.227481, -3.2405129, 0.980818]
            #
            # # 0.01
            # m1 = [2.20217881, -2.48587620, 0.984505]
            # m2 = [2.95998001, -3.03071600, 0.981375]
            # m3 = [2.94740544, -3.23671490, 0.981389]
            # m4 = [2.45126029, -3.25310720, 0.985082]
            # m5 = [2.23458811, -3.34167600, 0.980818]

            m = [m1, m2, m3, m4, m5]  # lame!

            # Fit_R2P = 1/(1+np.exp(-1*(m[file_values.shape[1]-1][0])*(P/file_values.shape[0] - (m[file_values.shape[1]-1][1]))))
            try:
                Fit_R2P = 1 / (1 + np.exp(-1 * (m[self.X.shape[1] - 1][0]) * (np.log(P / self.X.shape[0]) - (m[self.X.shape[1] - 1][1]))))

            except IndexError:

                raise IndexError('Fit 模式暂不支持5阶以上, 即自变量超过5个以上的数据, 继续分析请调整至CV模式')
            # Fit_R2P = (441.863117*24.915459*np.exp(0.05681148*P) / (441.863117 + 24.915459*(np.exp(0.05681148*P) - 1))) / file_values.shape[0]

            # cv_pred_R2 = cv_stored['y_pred_R2']
            # cv_pred_R2_ = []
            # for n in range(self.max_iter-4):
            #     cv_pred_R2_.append(np.mean(cv_pred_R2[n:n+4]))
            # cv_pred_R2_ = np.array(cv_pred_R2_)
            # Each_R2P = cv_pred_R2_

            # Fit_R2P_ = np.insert(Fit_R2P[:-1], 0, 0
            # fit_pred_R2_ = np.insert(fit_pred_R2[:-1], 0, 0)
            # Each_R2P = fit_pred_R2 - Fit_R2P
            # Each_R2P = (fit_pred_R2 - fit_pred_R2_) > (Fit_R2P - Fit_R2P_)  # δ_fit > δ_logistic
            # fit_P=0
            # for n in range(self.max_iter-1):
            #     if Each_R2P[n] > Each_R2P[n+1]:
            #         fit_P = n
            #         break

            # Each_R2P = (fit_pred_R2 - fit_pred_R2_) > (Fit_R2P - Fit_R2P_) * (-20*fit_pred_R2 + 21)  # δ_fit > δ_logistic * (-20*fit_R + 21),  ← cv_P
            Each_R2P = fit_pred_R2 / Fit_R2P

            cv_pred_R2 = cv_stored['y_pred_R2']
            cv_floor = P * (1 / (int(self.X.shape[0] ** 0.6) * self.X.shape[1]))
            cv_use = cv_pred_R2 > cv_floor

            p_df = pd.DataFrame({'cv_use': cv_use, 'cv_pred_R2': cv_pred_R2})
            largest_10p = list(p_df.loc[p_df['cv_use']].sort_values(by='cv_pred_R2', ascending=False).index[:5])
            # Each_R2P_minus = fit_pred_R2 - Fit_R2P
            # Each_R2P_prod = fit_pred_R2 / Fit_R2P
            #
            # Each_R2P_minus = Each_R2P_minus.argsort().argsort()
            # Each_R2P_prod = Each_R2P_prod.argsort().argsort() * (3/file_values.shape[1])
            # Each_R2P = Each_R2P_minus + Each_R2P_prod

            # fig = plt.figure()
            # plt.scatter(P, fit_pred_R2, color='blue', label='fit', alpha=0.5, marker='o',
            #             s=15)
            # plt.scatter(P, Fit_R2P, color='black', label='0.01pred', alpha=0.5, marker='o',
            #             s=15)
            # plt.scatter(P, Each_R2P, color='red', label='fit-0.01pred', alpha=0.5, marker='o',
            #             s=15)
            # cv_plus_p = cv_stored['p'] + 10
            # if cv_plus_p > self.max_iter:
            #     cv_plus_p = self.max_iter-1

            # Each_R2P = Each_R2P[:cv_stored['p']+1] # ← CV

            Each_R2P = [Each_R2P[cv_p] for cv_p in largest_10p]  # ← CV

            try:
                fit_P = largest_10p[np.argmax(Each_R2P)]

            except ValueError:
                fit_P = 0
            except TypeError:
                fit_P = 0
            # fit_P = np.where(Each_R2P)[0][-1] if Each_R2P.any() else cv_stored['p']

            # for n in range(cv_stored['p'], 0, -1):  # 从最后一位开始, 倒序遍历
            #     if (fit_pred_R2[n] - fit_pred_R2[n - 1]) > (16 - 15 * fit_pred_R2[n - 1]) / file_values.shape[0]:
            #         fit_P = n
            #         break

            # for n in range(cv_stored['p'], 0, -1):  # 从最后一位开始, 倒序遍历
            #     if (fit_pred_R2[n] - fit_pred_R2[n - 1]) > (16 - 15 * fit_pred_R2[n - 1]) / file_values.shape[0]:
            #         fit_P = n
            #         break

            # # Fit_R2P = (464.827202 * 34.3845124 * np.exp(0.05428937 * P) / (464.827202 + 34.3845124 * (np.exp(0.05428937 * P) - 1))) / file_values.shape[0]
            # # Fit_R2P = (fit_pred_R2 - np.array(Zero_p['AVERAGE'])[:self.max_iter])
            # # Each_R2P = Fit_R2P / fit_pred_R2
            #
            # # Mean_R2P_56 = (1/file_values.shape[0]) * np.power(P, 1.39435942)
            # # Each_R2P_56 = fit_pred_R2 / Mean_R2P_56
            # # Each_R2P[:46] = Each_R2P_56[:46]
            #
            # fit_P = np.argmax(Each_R2P)+2
            fit_R2 = fit_pred_R2[fit_P]
            storager['R2'] = fit_R2
            storager['p'] = fit_P
            storager['coef'] = fit_stored['p_arys'][0][:, fit_P].reshape(-1, 1)

            storager['fit_p'] = fit_P
            storager['fit_R2'] = fit_R2

            storager['y_preds'] = fit_y_preds
            storager['y_pred'] = fit_y_preds[:, fit_P].reshape(-1, 1)
            storager['y_pred_R2'] = fit_pred_R2

        elif self.fit_mode == 'Fit':

            fit_stored = self._DPLS_PLS_core(self.X, self.y, cv=1)

            if np.any(np.isnan(fit_stored['R2'])):
                return fit_stored

            fit_pred_R2 = fit_stored['y_pred_R2']
            fit_y_preds = fit_stored['y_preds']

            P = np.arange(1, self.max_iter + 1)

            # 好美的调试 ----------------------------------------------------------------------------------------------------------

            # average
            m1 = [2.21169467, -2.3817057, 0.984505]
            m2 = [3.02072028, -2.9587557, 0.981375]
            m3 = [2.99084683, -3.1657871, 0.981389]
            m4 = [2.44274849, -3.164627, 0.985082]
            m5 = [2.227481, -3.2405129, 0.980818]

            m = [m1, m2, m3, m4, m5]  # lame!

            try:
                Fit_R2P = 1 / (1 + np.exp(-1 * (m[self.X.shape[1] - 1][0]) * (np.log(P / self.X.shape[0]) - (m[self.X.shape[1] - 1][1]))))

            except IndexError:

                raise IndexError('Fit 模式暂不支持5阶以上, 即自变量超过5个以上的数据, 继续分析请调整至CV模式')

            Each_R2P = fit_pred_R2 / Fit_R2P
            fit_P = np.argmax(Each_R2P)

            fit_R2 = fit_pred_R2[fit_P]

            storager['R2'] = fit_R2
            storager['p'] = fit_P
            storager['coef'] = fit_stored['p_arys'][0][:, fit_P].reshape(-1, 1)

            storager['fit_p'] = fit_P
            storager['fit_R2'] = fit_R2

            storager['y_preds'] = fit_y_preds
            storager['y_pred'] = fit_y_preds[:, fit_P].reshape(-1, 1)
            storager['y_pred_R2'] = fit_pred_R2


        return storager

    @staticmethod
    def _transform_tuple_of_dicts(tuple_of_dicts):

        """
        将元组中的字典转换为新的结构。

        参数:
        tuple_of_dicts (tuple): 包含多个字典的元组，字典具有相同的键

        (dict{key_1:value_11, key_2:value_21, ...}, dict{key_1:value_12, key_2:value_22, ...})
        ↓
        (dict{key_1:[value_11, value_12], key_2:[value_21,key_2:value_22, ...})

        返回:
        dict: 新结构的字典，形式为 {'key': (value1, value2, ..., valueN)}。
        """

        # 初始化结果字典
        result_dict = {}

        # 遍历元组中的每个字典
        for d in tuple_of_dicts:
            for key, value in d.items():
                # 如果键不在结果字典中，初始化一个空列表
                if key not in result_dict:
                    result_dict[key] = []
                # 将当前字典的值添加到对应键的列表中
                result_dict[key].append(value)

        # 将列表转换为元组
        for key in result_dict:
            result_dict[key] = tuple(result_dict[key])

        return result_dict


    def _fit(self, X, y, fit_mode: Literal['CV', 'Fit', 'Fit_rectify'] = 'Fit_rectify', **kwargs):

        """

        X_ : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of train_samples
            and `n_features` is the number of features.

        y_ : Ignored.

        :param X:
        :param y:

        :return:
        """

        if self.R_mode == 'fusion':

            stored_data = [self._fit_rectify(X=X, y=y, fit_mode=fit_mode, **kwargs)]

        elif self.R_mode == 'single':

            if self.n_jobs == 1:
                stored_data = []

                for i in tqdm(range(X.shape[1]), desc=f"DPLS_single",
                              leave=True,
                              ncols=88,
                              colour='white', disable=not self.bar):
                    data_i = self._fit_rectify(X=X[:, i].reshape(-1, 1), y=y, fit_mode=fit_mode, **kwargs)
                    stored_data.append(data_i)

            else:
                stored_data = Parallel(n_jobs=1)(
                    delayed(self._fit_rectify)(X=X[:, i].reshape(-1, 1), y=y, fit_mode=fit_mode, **kwargs) for i
                    in range(X.shape[1]))
                # n_jobs暂时固定为1, 多线程显示有不能打包的参数 (可能存在循环引用问题?)

        else:
            raise ValueError('R_mode must be either "fusion" or "single"')

        stored_dict = self._transform_tuple_of_dicts(stored_data)

        self.__dict__.update(stored_dict)

        return self


    def fit(self, X, y, fit_mode: Literal['CV', 'Fit', 'Fit_rectify'] = 'Fit_rectify',intercept=False, **kwargs):

        X = test_tools_v312.to_2D_ary(X)
        y = test_tools_v312.to_2D_ary(y)

        if self.whiten:
            X = test_tools_v312.stdize(X)

        self.X = X
        self.y = y

        return self._fit(X=X, y=y, fit_mode=fit_mode, **kwargs)


    def fit_transform(self, X, y, fit_mode: Literal['CV', 'Fit', 'Fit_rectify'] = 'Fit', **kwargs):

        if self.R_mode == 'fusion':
            return self.fit(X=X, y=y, fit_mode=fit_mode, **kwargs).y_pred[0]
        else:
            return np.hstack(self.fit(X=X, y=y, fit_mode=fit_mode, **kwargs).y_pred)


    def _predict(self, X_test, X_train, coef, **kwargs):

        X_test = test_tools_v312.to_2D_ary(X_test)
        X_train = test_tools_v312.to_2D_ary(X_train)

        X = np.vstack([X_train, X_test])

        train_sample_num = X_train.shape[0]

        all_sample_distance = DPLS_distance(X, distance_pattern=self.distance_pattern, **kwargs)

        if self.square:
            all_sample_distance = all_sample_distance @ all_sample_distance.T
        else:
            pass

        y_pred = all_sample_distance[:, :train_sample_num] @ coef

        return y_pred[train_sample_num:, 0]

    def predict(self, X_test, **kwargs):

        X_test = test_tools_v312.to_2D_ary(X_test)

        if self.whiten:
            X_test = test_tools_v312.stdize(X_test)

        if not self.coef:
            raise NotImplementedError('No coef, fit first')

        if X_test.shape[1] != self.X.shape[1]:
            raise ValueError('X_test and X_train must have same n_features')

        if self.R_mode == 'fusion':

            y_pred = self._predict(X_test=X_test, X_train=self.X, coef=self.coef[0], **kwargs)

        else:

            y_pred_list = []

            for i, coef in enumerate(self.coef):
                y_pred_i = self._predict(X_test=X_test[:, i], X_train=self.X[:, i], coef=coef, **kwargs)
                y_pred_list.append(y_pred_i)

            y_pred = np.hstack(y_pred_list)

        return y_pred

    def _graph(self, X: np.ndarray):

        DPLS_graph = pd.DataFrame(columns=list(range(1, X.shape[1] + 1)), dtype='float64')

        def calculate_graph(i_):

            X_i_as_Y_ = DPLS(R_mode='single', bar=False, n_jobs=1, fit_rectify=self.fit_rectify).fit(X=X,
                                                                                                     y=X[:, i_].reshape(
                                                                                                         -1, 1)).R2

            return X_i_as_Y_

        if self.n_jobs == 1:

            for i in tqdm(range(X.shape[1]), desc=f"正在构建DPLS_graph",
                          leave=True,
                          ncols=88,
                          colour='white', disable=not self.bar):
                DPLS_graph.loc[i + 1] = calculate_graph(i)

        else:

            graph_result = Parallel(n_jobs=self.n_jobs)(
                delayed(calculate_graph)(i) for i in tqdm(range(X.shape[1]), desc=f"正在构建DPLS_graph",
                                                          leave=True,
                                                          ncols=88,
                                                          colour='white', disable=not self.bar))

            for t, X_i_as_Y in enumerate(graph_result):
                DPLS_graph.loc[t + 1] = X_i_as_Y

        return DPLS_graph

    # graph: 求 file_values 两两之间(一个X做自变量, 另一个X做因变量)的 R2
    def graph(self, X):
        X = test_tools_v312.to_2D_ary(X)
        DPLS_graph = self._graph(X)
        return DPLS_graph

    def _pairs(self, X: np.ndarray, y: np.ndarray):

        DPLS_pairs = pd.DataFrame(columns=list(range(1, X.shape[1] + 1)), dtype='float64')

        def calculate_pairs(i_):

            X_split = np.split(X, X.shape[1], axis=1)
            X_i_as_left = [np.hstack([X_split[i_], X_i]) for X_i in X_split]
            X_i_as_left_R = []

            for X_i in X_i_as_left:
                X_i_r = DPLS(R_mode='fusion', distance_pattern=['Euc'], n_jobs=1).fit(X=X_i, y=y).R2[0]
                X_i_as_left_R.append(X_i_r)

            return X_i_as_left_R

        if self.n_jobs == 1:

            for i in tqdm(range(X.shape[1]), desc=f"正在构建DPLS_pairs",
                          leave=True,
                          ncols=88,
                          colour='white', disable=not self.bar):
                DPLS_pairs.loc[i + 1] = calculate_pairs(i)

        else:

            DPLS_pairs_para = Parallel(n_jobs=self.n_jobs)(
                delayed(calculate_pairs)(i) for i in tqdm(range(X.shape[1]), desc=f"正在构建DPLS_pairs",
                                                          leave=True,
                                                          ncols=88,
                                                          colour='white',
                                                          disable=not self.bar))

            for t, para_result in enumerate(DPLS_pairs_para):
                DPLS_pairs.loc[t + 1] = para_result

        return DPLS_pairs

    # 求 file_values 两两配对为 pair 时, 所有 pairs 与 y 的 DPLS_R
    def pairs(self, X: np.ndarray, y: np.ndarray):
        DPLS_pairs = self._pairs(X, y)
        return DPLS_pairs


# if __name__ == '__main__':
#
#     from Projects.Muti_func_creat.muti_func_test import gen_y_exp, add_noise
#
#
#     for i in range(5):
#
#         x,file_values,y_exp,x_picked = gen_y_exp(sample_num=1000, param_num=3, use_x_num=3, x_seed=i)
#
#         # x['x_2'] = np.sin(np.pi * x['x_1'])
#         # aa = add_noise(x['x_2'], 1.0)
#         # print('R:x_2,x_2+noise',test_tools_v312.calculate_corr(aa, x['x_2']))
#         # x['x_2'] = aa
#
#         # file_values = pd.DataFrame()
#         # file_values['sin(x_1)'] = np.sin(x['x_1'])
#         # file_values['x_1*x_2'] = x['x_1']
#         # file_values['x_1'] = x['x_1']
#         # file_values['x_1^2'] = np.power(x['x_1'], 2)
#         # file_values['sin(x_2)'] = np.sin(x['x_2'])
#
#         # y_exp = np.sum(file_values.to_numpy(), axis=1)
#
#         y_create_obs = add_noise(y_exp, 2.05, func_seed=None)
#         print('R:y_exp,yobs', test_tools_v312.calculate_corr(y_exp, y_create_obs))
#
#         Euc_graph = DPLS(R_mode='fusion', max_iter=50, whiten=True, tol=1e-5, cv=5, eig_solver='sklearn', distance_pattern=['Euc'], transpose=False, square=True).graph(x)
#         a=1



    # y_pred_Rs = []
    #
    # x_l, X_l, y_exp_l, x_picked_l = gen_y_exp(sample_num=1001, param_num=1, use_x_num=1, x_seed=1, linear=True)
    # x_nl, X_nl, y_exp_nl, x_picked_nl = gen_y_exp(sample_num=1001, param_num=1, use_x_num=0, x_to_x_num=1, x_seed=1,
    #                                               func_seed=5)

    # x['x_2'] = np.sin(np.pi * x['x_1'])
    # aa = add_noise(x['x_2'], 1.0)
    # print('R:x_2,x_2+noise',test_tools_v312.calculate_corr(aa, x['x_2']))
    # x['x_2'] = aa

    # file_values = pd.DataFrame()
    # file_values['sin(x_1)'] = np.sin(x['x_1'])
    # file_values['x_1*x_2'] = x['x_1']
    # file_values['x_1'] = x['x_1']
    # file_values['x_1^2'] = np.power(x['x_1'], 2)
    # file_values['sin(x_2)'] = np.sin(x['x_2'])

    # y_exp = np.sum(file_values.to_numpy(), axis=1)




    #
    # y_obs_l = add_noise(y_exp_l, 5, func_seed=None)
    # y_obs_nl = add_noise(y_exp_nl, 5, func_seed=None)
    #
    # print('R:y_exp_l,yobs', test_tools_v312.calculate_corr(y_exp_l, y_obs_l))
    # print('R:y_exp_nl,yobs', test_tools_v312.calculate_corr(y_exp_nl, y_obs_nl), '\n')
    #
    # l_obj = DPLS(R_mode='fusion', max_iter=50, whiten=True, tol=1e-5, cv=5, eig_solver='sklearn',
    #              distance_pattern=['Euc'], transpose=False, square=False).fit(x_l, y_obs_l)
    # nl_obj = DPLS(R_mode='fusion', max_iter=50, whiten=True, tol=1e-5, cv=5, eig_solver='sklearn',
    #               distance_pattern=['Euc'], transpose=False, square=False).fit(x_nl, y_obs_nl)
    #
    # l_y_pred = l_obj.y_pred[0]
    # nl_y_pred = nl_obj.y_pred[0]
    #
    # print('PersonR_xy[l_pred, y_exp_l]: ', test_tools_v312.calculate_corr(l_y_pred, y_exp_l))
    # print('PersonR_xy[nl_pred, y_exp_nl]: ', test_tools_v312.calculate_corr(nl_y_pred, y_exp_nl), '\n')
    #
    # print('PersonR_xy[l_pred, y_obs_l]: ', test_tools_v312.calculate_corr(l_y_pred, y_obs_l))
    # print('PersonR_xy[nl_pred, y_obs_nl]: ', test_tools_v312.calculate_corr(nl_y_pred, y_obs_nl), '\n')
    #
    # print('PersonR_xy[l_pred, x_l]: ', test_tools_v312.calculate_corr(x_l, l_y_pred))
    # print('PersonR_xy[nl_pred, x_nl]: ', test_tools_v312.calculate_corr(x_nl, nl_y_pred), '\n')
    #
    # l_obj = DPLS(R_mode='fusion', max_iter=50, whiten=True, tol=1e-5, cv=5, eig_solver='sklearn',
    #              distance_pattern=['Euc'], transpose=False, square=False).fit(y_obs_l, x_l)
    # nl_obj = DPLS(R_mode='fusion', max_iter=50, whiten=True, tol=1e-5, cv=5, eig_solver='sklearn',
    #               distance_pattern=['Euc'], transpose=False, square=False).fit(y_obs_nl, x_nl)
    #
    # l_y_pred = l_obj.y_pred[0]
    # nl_y_pred = nl_obj.y_pred[0]
    #
    # print('- reverse ----------------------------------------------------------------------------\n')
    #
    # print('PersonR_yx[l_pred, x_l]: ', test_tools_v312.calculate_corr(l_y_pred, x_l))
    # print('PersonR_yx[nl_pred, x_nl]: ', test_tools_v312.calculate_corr(nl_y_pred, x_nl), '\n')
    #
    # print('PersonR_yx[l_pred, y_obs_l]: ', test_tools_v312.calculate_corr(y_obs_l, l_y_pred))
    # print('PersonR_yx[nl_pred, y_obs_nl]: ', test_tools_v312.calculate_corr(y_obs_nl, nl_y_pred), '\n')






    # y_pred_df = pd.DataFrame(np.array(y_pred_Rs).T)
    # y_pred_df.to_excel('500sample_PLS_4x_p_250.xlsx')
    #
    # Mah_obj = DPLS(R_mode='fusion', max_iter=50, whiten=True, distance_pattern=['Mah']).fit(x.iloc[:500, :], y_create_obs[:500])
    # Single_obj = DPLS(R_mode='single', max_iter=50, whiten=True).fit(x.iloc[:500, :], y_create_obs[:500])
    #
    # print('Euc_DPLSR:x,y',Euc_obj.R2)
    # print('Mah_DPLSR:x,y',Mah_obj.R2)
    # print('Single_DPLSR:x,y',Single_obj.R2)
    #
    # print('R:DPLS_pred,y_create_obs',test_tools_v312.calculate_corr(obj.y_pred[0], y_create_obs))
    # print('R:DPLS_pred,y_exp',test_tools_v312.calculate_corr(obj.y_pred[0], y_exp))
