from itertools import combinations
from typing import Literal, Iterable, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from tqdm import tqdm

from GUI_functions import test_tools_v312
from DPLS_GUI import DPLS
from GUI_functions.muti_func_test import gen_y_exp, gen_seed

from stqdm import stqdm

Distance_Option = Literal['Mah', 'Euc', 'Pairs', 'Ming', 'origin']
Distance_Option_Iterable = Union[Distance_Option, Iterable[Distance_Option]]


class DPLS_Checker:

    def __init__(self,

                 x_seed: range | list | int = range(5, 10),
                 func_seed: range | list | int = range(20, 25),
                 test_seed: int = 10086,

                 max_iter: int = 20,
                 tol: float = 1e-3,
                 square: bool = True,
                 eig_solver: Literal['pow', 'sklearn'] = 'pow',
                 distance_pattern: Distance_Option_Iterable = 'Mah',

                 fit_mode:Literal["Fit", "CV", "Fit_rectify"]="Fit_rectify",
                 keep_irrelevant=False,
                 redundancy=False,
                 transpose=False,

                 train_samples: int = 500,
                 test_samples: int = 0,
                 check_single: bool = False,
                 dots_num: int = 30,

                 use_x_num: int | list | range = 10,
                 x_num: int | list | range = 1,
                 x_to_x_num: int | list | range = 1,
                 x_to_x_level: int | list | range = 3,

                 cv: int = 5,

                 x_start=-1,
                 x_end=1,

                 thread: int = 1,
                 whiten: bool = True,
                 core: Literal['JJ'] = 'JJ',
                 noise_start=np.append(np.arange(0, 3, 0.02), np.arange(3, 8, 0.1)),

                 input_func_dict:dict | None = None,

                 **kwargs,

                 ):

        """

    | seed 类参数 ------------------------------------------------------------------------------------------------------

        :param x_seed: 随机种子, 用于随机生成在 [x_start, x_end] 区间内自变量的值
        :param func_seed: 随机种子, 用于随机生成 x 的方程
        :param test_seed: 随机种子, 用于随机生成噪音

        所有的随机种子支持两种输入: int | list

        |例 1:

            x_seed: [0,1,2,3]
            func_seed:[4,5,6,7]
            test_seed:[8,9,10,11]

            程序会检查 4个 由 func_seed:[4,5,6,7] 决定的方程, 这些方程的自变量由种子 x_seed: [0,1,2,3] 决定, 程序对每个
            方程的 dots_num 次测试所添加的不同程度的噪音由测试种子 test_seed:[8,9,10,11] 决定

        |例 2:

            x_seed:[0,1,2,3]
            func_seed:5
            test_seed:6

            程序会检查 4个 [相同] 的方程 由 func_seed:5 决定, 这些 [相同] 方程的自变量 [不同], 由种子 x_seed: [0,1,2,3] 决定, 程序对每个
            方程的 dots_num 次测试所添加的不同程度的噪音都 [相同], 由测试种子 test_seed:6 决定

        |例 3:

            x_seed:1
            func_seed:5
            test_seed:7

            程序会检查 1 个方程 由 func_seed:5 决定, 此方程的自变量由种子 x_seed: 1 决定, 程序对该
            方程的 dots_num 次测试所添加的不同程度的噪音由测试种子 test_seed:7 决定


    ------------------------------------------------------------------------------------------------------------------

        :param max_iter: PLS 最大迭代层
        :param tol: PLS 迭代收敛标准, 两次迭代间残差模长变化率小于 tol 则判定收敛
        :param square: 是否让距离矩阵规方, 规方的距离矩阵 = 距离矩阵 * 距离矩阵.T
        :param eig_solver: 选择PLS计算的核心, sklearn:sk-learn写的,  Pow: 我写的 (没人家快 😅

        :param distance_pattern: 距离矩阵模式

            假设 X 形状 : 100*5

            | 'Euc': 欧氏距离, 形状 100*100
            | 'Mah': 曼哈顿距离, 形状 100*500
            | 'Pairs': 两两距离的组合, 形状 100*1500
            | 'Ming': 明氏距离, 形状 100*100

            distance_pattern 输入形式要求为 list['mode1', 'mode2', ...]

            举例: 当输入 distance_pattern = ['tensor'], 则生成曼哈顿矩阵
                 当输入 distance_pattern = ['tensor', 'vector'], 则生成 [曼哈顿矩阵]和[欧式距离矩阵] 按 mode 顺序拼接而成的矩阵

        :param fit_mode: Fit_rectify: fit矫正, CV: 不矫正, 使用 cv 结果, Fit: 不矫正, 使用 Fit 结果
        :param keep_irrelevant: 开发中的功能
        :param redundancy: 开发中的功能

        :param train_samples: check用的样本数, 可以理解为训练集
        :param test_samples: 测试集样本数, 可以为 0
        :param check_single: bool, 是否返回每个 x 单独的和 y 的 DPLS_R, True: 返回, 值存储在 self.X_R 中


    | num 类参数 ------------------------------------------------------------------------------------------------------

        所有的num类参数支持两种输入: int | list, 规则同seed类

        :param dots_num: 每个方程测试的次数
        :param use_x_num: 生成 x 的上限个数, 但不一定会全部使用
        :param x_num: 独立方程的个数, 独立方程: f(x)
        :param x_to_x_num:互作方程的个数, 互作方程: f(x_1*x_2...x_n), n: 最大为 x_to_x_level
        :param x_to_x_level: 互作方程的上限阶数, 但不一定会取到最大

        所有的 num 类形参支持两种输入: int | list, 适用方法与 seed 类参数类似

    ------------------------------------------------------------------------------------------------------------------

        :param cv: 独立测试折数

        :param x_start: 自变量 x 取值下限
        :param x_end: 自变量 x 取值上限

        :param thread: 线程数, 用于多线程功能
        :param whiten: bool, True: 标准化, False: 不标准化
        :param core: 开发中功能
        :param noise_start: 用于拟合不同噪音强度的 level

        :param kwargs: 其他参数

        """

        #   | 参数赋值 --------------------------------------------------------------------------------------------------------

        self.x_seed = x_seed
        self.func_seed = func_seed
        self.test_seed = test_seed

        self.max_iter = max_iter
        self.tol = tol
        self.square = square
        self.transpose = transpose
        self.eig_solver = eig_solver
        self.fit_mode = fit_mode
        self.check_single = check_single

        self.keep_irrelevant = keep_irrelevant
        self.redundancy = redundancy

        self.dots_num = dots_num
        self.train_samples = train_samples
        self.test_samples = test_samples

        self.use_x_num = use_x_num
        self.x_num = x_num
        self.x_to_x_num = x_to_x_num
        self.x_to_x_level = x_to_x_level

        self.cv = cv
        self.R_threshold = 1
        self.x_start = x_start
        self.x_end = x_end

        self.thread = thread
        self.whiten = whiten
        self.core = core
        self.noise_start = noise_start
        self.distance_pattern = distance_pattern

        self.func_info = {} if not input_func_dict else input_func_dict
        self.func_input = False if not input_func_dict else True

        self.R_real_start = pd.DataFrame(index=self.noise_start)
        self.seed_start = gen_seed(len(noise_start), rand_seed=self.test_seed)[0]



        #   | 检查参数类型, 以及检查参数值是否合法 ----------------------------------------------------------------------------------

        if not self.func_input:

            check_param_list = ['x_seed', 'func_seed', 'use_x_num', 'x_num', 'x_to_x_num', 'x_to_x_level']

            param_lens = set(
                len(self.__dict__[x]) if isinstance(self.__dict__[x], list | range) else 1 for x in check_param_list)

            if len(param_lens) == 1:
                check_len = param_lens.pop()

            elif len(param_lens) == 2 and 1 in param_lens:
                check_len = param_lens.pop()
                if check_len == 1:
                    check_len = param_lens.pop()

            else:
                raise ValueError('类型为 [int | list] 的参数必须为 int 或全同长度的 list')

            for param in check_param_list:

                if isinstance(self.__dict__[param], int):
                    self.__dict__[param] = [self.__dict__[param]] * check_len

            if test_samples < 0:
                raise ValueError('形参 test_samples 的实参值必须为 >= 0 的 int ')

        else:

            check_len = len(self.func_info)

        #   | 生成check用方程 --------------------------------------------------------------------------------------------------

        for i in range(check_len):

            if not self.func_input:

                x, X, y_exp, x_picked = gen_y_exp(sample_num=(self.train_samples + self.test_samples),
                                                  use_x_num=self.use_x_num[i],
                                                  param_num=self.use_x_num[i],
                                                  max_iter=self.max_iter,
                                                  redundancy=self.redundancy,

                                                  x_num=self.x_num[i],
                                                  x_to_x_num=self.x_to_x_num[i],
                                                  func_seed=self.func_seed[i],
                                                  x_seed=self.x_seed[i],
                                                  x_to_x_level=self.x_to_x_level[i],
                                                  x_start=self.x_start,
                                                  x_end=self.x_end,

                                                  **kwargs)

                #   | 自定义方程区 -----------------------------------------------------------------------------------------------------

                # x_picked = ['x_2', 'x_3']
                # x['x_2'] = np.sin(np.pi * x['x_1'])
                # aa = add_noise(x['x_2'], 1.0)
                # print('R:x_2,x_2+noise', test_tools_v312.calculate_corr(aa, x['x_2']))
                # x['x_2'] = aa
                #
                # X = pd.DataFrame()
                # # X['x_1+x_2'] = x['x_1'] + x['x_2']
                # # X['x_1'] = x['x_1']
                # X['x_1^2'] = np.power(x['x_1'], 2)
                # X['sin(x_2)'] = np.sin(x['x_2'])
                #
                # y_exp = np.sum(X.to_numpy(), axis=1)

                #   | 构建方程名 -------------------------------------------------------------------------------------------------------

                func_name = 'y=' + '+'.join(list(X.columns)) + f'[{self.func_seed[i]}]'
                if len(func_name) > 75:
                    func_name = func_name[:75] + ' ...'

                #   | 记录方程至 self.func_info ----------------------------------------------------------------------------------------

                self.func_info[i] = {
                    'x': x,
                    'X': X,
                    'func_name': func_name,
                    'x_picked': x_picked,
                    'y_exp': y_exp,
                }

            else:

                y_exp = self.func_info[i]['y_exp']
                func_name = self.func_info[i]['func_name']

            #   | 拟合 noise_start ------------------------------------------------------------------------------------------------

            y_obs_start = []
            for l, level in enumerate(self.noise_start):
                np.random.seed(self.seed_start[l])
                y_obs = y_exp + np.random.normal(size=y_exp.shape[0], loc=0, scale=y_exp.std() * level)
                y_obs_start.append(y_obs.reshape(-1, 1))

            y_obs_start = np.hstack(y_obs_start)
            self.R_real_start[func_name] = test_tools_v312.calculate_corr(y_obs_start, y_exp)

        degree = 2
        coefficients = np.polyfit(self.R_real_start.mean(axis=1), self.noise_start, degree)
        self.polynomial = np.poly1d(coefficients)

        #   | 记录方程名至 self.func_names -------------------------------------------------------------------------------------
        if self.func_input:

            self.func_names = {f: self.func_info[f]['func_name'] for f in list(self.func_info.keys())}

        else:

            self.func_names = {f'{f}-{func_seed_}': self.func_info[f]['func_name']
                               for f, func_seed_ in enumerate(self.func_seed)}

        #   | 创建噪音种子 self.dots_seed --------------------------------------------------------------------------------------

        self.dots_seed = gen_seed(param_num=dots_num, rand_seed=self.test_seed)[0]

        #   | 创建计算过程属性容器 -----------------------------------------------------------------------------------------------

        self.R_true_DF = pd.DataFrame(dtype='float64')
        self.R_return_DF = pd.DataFrame(dtype='float64')
        self.R_pred_DF = pd.DataFrame(dtype='float64')
        self.P_return_DF = pd.DataFrame(dtype='float64')
        self.X_R = []
        self.P_R = []

    def _x_equity(self,
                  test_levels,
                  **kwargs,
                  ):

        def check_f(f):

            pbar_check = stqdm(desc=f"-Checking [{self.func_names[f]}] equity",
                              leave=True,
                              total=self.dots_num,
                              ncols=80,
                              colour='white')

            X_R = {}
            R_true_Dict = {}
            R_return_Dict = {}
            R_pred_Dict = {}

            P_return_Dict = {}
            P_R = {}

            x = self.func_info[f]['x']
            x_picked = self.func_info[f]['x_picked']
            print(x_picked)
            func_name = self.func_info[f]['func_name']
            y_exp = self.func_info[f]['y_exp']

            for i, level in enumerate(test_levels):

                level = abs(level)

                np.random.seed(self.dots_seed[i])
                y_obs = y_exp + np.random.normal(size=y_exp.shape[0], loc=0, scale=y_exp.std() * level)

                if self.keep_irrelevant:
                    x_use = x
                else:
                    x_use = x[x_picked]
                    x_use = test_tools_v312.to_2D_ary(x_use)

                x_train_fusion = DPLS(dtype=float, tol=self.tol, transpose=self.transpose,
                                      whiten=self.whiten, square=self.square,
                                      distance_pattern=self.distance_pattern,
                                      cv=self.cv,
                                      eig_solver=self.eig_solver,
                                      max_iter=self.max_iter, **kwargs).fit(x_use[:self.train_samples, :],
                                                                            y_obs[:self.train_samples],
                                                                            fit_mode=self.fit_mode,
                                                                            R_mode='fusion')

                if self.check_single:
                    x_train_single = DPLS(dtype=float, R_mode='single', tol=self.tol,
                                          whiten=self.whiten, square=self.square,
                                          distance_pattern=self.distance_pattern,
                                          cv=self.cv,
                                          eig_solver=self.eig_solver,
                                          max_iter=self.max_iter, bar=False, **kwargs).fit(
                        x_use[:self.train_samples, :], y_obs[:self.train_samples], fit_mode=self.fit_mode)

                    X_R[level] = x_train_single.R2

                R2_return = x_train_fusion.R2[0]
                R2_True = np.corrcoef(y_exp.flatten(), y_obs.flatten())[0, 1] ** 2
                P_return = x_train_fusion.p[0]

                if self.test_samples > 0:
                    y_pred = x_train_fusion.predict(x_use[self.train_samples:, :])
                    R2_pred = test_tools_v312.calculate_corr(y_pred, y_obs[self.train_samples:])[0]
                else:
                    R2_pred = None

                R_true_Dict[level] = R2_True
                R_return_Dict[level] = R2_return
                R_pred_Dict[level] = R2_pred
                P_return_Dict[level] = P_return
                P_R[level] = x_train_fusion.y_pred_R2[0]
                pbar_check.update(1)

            R_true_DF = pd.DataFrame.from_dict({f'{func_name}': R_true_Dict})
            R_return_DF = pd.DataFrame.from_dict({f'{func_name}': R_return_Dict})
            R_pred_DF = pd.DataFrame.from_dict({f'{func_name}': R_pred_Dict})
            X_R = pd.DataFrame.from_dict(X_R)
            P_R = pd.DataFrame.from_dict(P_R)
            P_return_DF = pd.DataFrame.from_dict({f'{func_name}': P_return_Dict})

            storager = {

                'R_true_DF': R_true_DF,
                'R_return_DF': R_return_DF,
                'R_pred_DF': R_pred_DF,
                'X_R': X_R,
                'P_R': P_R,
                'P_return_DF': P_return_DF,

            }

            return storager


        if self.func_input:

            para_result = Parallel(n_jobs=self.thread)(delayed(check_f)(f) for f in list(self.func_info.keys()))

        else:

            para_result = Parallel(n_jobs=self.thread)(delayed(check_f)(f) for f in range(len(self.func_seed)))

        self.R_true_DF = pd.concat([para_result_i['R_true_DF'] for para_result_i in para_result], axis=1)
        self.R_return_DF = pd.concat([para_result_i['R_return_DF'] for para_result_i in para_result], axis=1)
        self.R_pred_DF = pd.concat([para_result_i['R_pred_DF'] for para_result_i in para_result], axis=1)
        self.X_R = [para_result_i['X_R'] for para_result_i in para_result]
        self.P_R = [para_result_i['P_R'] for para_result_i in para_result]
        self.P_return_DF = [para_result_i['P_return_DF'] for para_result_i in para_result]

    def check_x_equity(self,
                       region: tuple = (1, 0),
                       plot=True,
                       plot_mode: Literal['each', 'entire'] = 'entire',
                       desc='',
                       notice='',
                       details=False,
                       **kwargs,
                       ):

        """

        :param region: R的取值范围, (1, 0)代表 R 的范围取到 [1, 0], (1, 0.6)代表 R 的范围取到 [1, 0.6], 不做噪声高于 0.4的测试
        :param plot: bool, True: 显示结果图, False: 不显示
        :param plot_mode: 结果图的展示方式, 'each': 每个方程输出一个结果图, 'entire': 所有方程在一个结果图内展示
        :param desc: 结果图的主标
        :param notice: 结果图的备注, 备注会展示在 details 区
        :param details: bool, True, 展示
        :param kwargs:
        :return:

        """

        self.R_threshold = region

        # 拟合曲线
        level_fit = np.linspace(region[0], region[1] + 0.1, self.dots_num)
        test_levels = self.polynomial(level_fit)

        self._x_equity(test_levels=test_levels, **kwargs)

        if plot:
            return self.plot(desc=desc, notice=notice, plot_mode=plot_mode,
                      x_df=(1 - self.R_true_DF), y_df=self.R_return_DF, details=details,
                      x_label='DPLS_R^2', y_label='Noise (1 - R<y_exp, y_obs>^2)')

        return self

    def _pairs_y_preds_equity(self,
                              test_levels,
                              y_perd_mode: Literal['PLS', 'distance_PLS'] = 'PLS',

                              **kwargs, ):

        def check_seed(f, func_seed):

            pbar_check = tqdm(desc=f"-Checking pairs_y_preds_equity",
                              leave=True,
                              total=self.dots_num,
                              ncols=80,
                              colour='white')

            func_R_true_DF = pd.DataFrame(index=self.noise_start, dtype='float64')
            func_R_return_DF = pd.DataFrame(index=self.noise_start, dtype='float64')
            func_R_pred_DF = pd.DataFrame(index=self.noise_start, dtype='float64')

            for i, level in enumerate(test_levels):

                np.random.seed(self.dots_seed[i])
                level = abs(level)

                x = self.func_info[f]['x']
                x_picked = self.func_info[f]['x_picked']
                func_name = self.func_info[f]['func_name']

                y_exp = self.func_info[f]['y_exp']
                y_obs = y_exp + np.random.normal(size=y_exp.shape[0], loc=0, scale=y_exp.std(axis=0) * level)

                np.random.seed(None)

                R2_True = np.corrcoef(y_exp.flatten(), y_obs.flatten())[0, 1] ** 2

                if self.keep_irrelevant:
                    x_use = x
                else:
                    x_use = x[x_picked]

                # 生成配对
                pairs = list(combinations(x_use.columns.tolist(), 2))
                pairs = [list(pair) for pair in pairs]

                # y_preds: y_pred 容器, y_preds = [y_pred1, y_pred2, ..., y_predn ]
                y_preds = []

                # 逐个 pair 求 y_pred
                for pair in pairs:
                    # 索引
                    x_pair = x_use[pair]

                    # 求 y_pred
                    y_pair_pred = DPLS(R_mode='fusion',
                                       cv=self.cv,
                                       distance_pattern=['Euc'],
                                       fit_mode=self.fit_mode,
                                       **kwargs).fit(x_pair, y_obs).y_pred[0]

                    # reshape and append
                    y_pair_pred = y_pair_pred.reshape(-1, 1)
                    y_preds.append(y_pair_pred)

                # hstack
                y_hat = np.hstack(y_preds)

                if y_perd_mode == 'PLS':

                    # PLS_fit
                    y_hat_pls = PLSRegression(n_components=y_hat.shape[1])
                    y_hat_pls.fit(y_hat, y_obs)

                    # PLS_predict
                    y_hat_pred = y_hat_pls.predict(y_hat)
                    y_hat_R = np.corrcoef(y_obs.flatten(), y_hat_pred.flatten())[0, 1] ** 2
                    return_R2 = y_hat_R

                elif y_perd_mode == 'distance_PLS':

                    # PLS_Euc_pred
                    y_obj = DPLS(R_mode='fusion', cv=1, distance_pattern=['origin'], fit_mode="Fit").fit(y_hat,
                                                                                                            y_obs)
                    return_R2 = y_obj.R2[0]

                else:
                    raise AttributeError('')

                func_R_true_DF.loc[level, func_name] = R2_True
                func_R_return_DF.loc[level, func_name] = return_R2

                pbar_check.update(1)

            return func_R_true_DF, func_R_return_DF

        para_result = Parallel(n_jobs=self.thread)(
            delayed(check_seed)(f, func_seed_i) for f, func_seed_i in enumerate(self.func_seed))

        Real_DF = pd.concat([df_A for df_A, _ in para_result], axis=0)
        return_DF = pd.concat([df_B for _, df_B in para_result], axis=0)

        return Real_DF, return_DF

    def check_pairs_to_ypreds_equity(self,
                                     threshold=1.0,
                                     y_perd_mode: Literal['PLS', 'distance_PLS'] = 'PLS',
                                     plot=True,
                                     plot_mode: Literal['each', 'entire'] = 'entire',
                                     desc='',
                                     notice='',
                                     **kwargs,
                                     ):

        # 拟合曲线
        level_fit = np.linspace(threshold + 0.1, -0.1, self.dots_num)
        test_levels = self.polynomial(level_fit)

        y_preds_R_real, y_preds_R_return = self._pairs_y_preds_equity(test_levels=test_levels, y_perd_mode=y_perd_mode,
                                                                      **kwargs)

        if plot:
            self.plot(desc=desc, notice=notice, plot_mode=plot_mode,
                      x_df=(1 - self.R_true_DF), y_df=self.R_return_DF,
                      x_label='DPLS_R^2', y_label='Noise (1 - R<y_exp, y_obs>^2)')

        return y_preds_R_real, y_preds_R_return


    def _plot(self, x_df, y_df,
              desc: str = '', x_label: str = '', y_label: str = '',
              details: bool = False, color: list = ('red'), notice=None,
              ):

        # Position & Face_color & Edge -------------------------------------------------------------------------------

        detail_extra_len = 0
        if details:
            detail_extra_len = 1.7

        fig = plt.figure(figsize=(8 + detail_extra_len, 6), dpi=500, facecolor='#fcfdfd')
        fig.patch.set_edgecolor('gray')  # 设置边框颜色
        fig.patch.set_linewidth(2)
        ax = plt.axes((0.085, 0.1, (7 - detail_extra_len) / 8, 7 / 9))

        # Scatter ---------------------------------------------------------------------------------------------------

        for i, feature in enumerate(y_df.columns):
            plt.scatter(x_df[feature], y_df[feature], color=color[i], label=feature, alpha=0.6, marker='o',
                        s=15)

        # Label -----------------------------------------------------------------------------------------------------

        plt.ylabel(f'{x_label}', fontdict={'family': 'Calibri'}, fontsize=11.5)
        plt.xlabel(f'{y_label}', fontdict={'family': 'Calibri'}, fontsize=11.5)
        plt.legend(fontsize='x-small')

        # Title -----------------------------------------------------------------------------------------------------

        title_1 = rf'sample:{self.train_samples}, [max_iter={self.max_iter}], distance:{self.distance_pattern}, domain[{self.x_start}, {self.x_end}]'
        title_2 = f'[MSE:{np.nansum((np.array(1 - x_df) - np.array(y_df)) ** 2) / x_df.shape[0]:.5f}]'  # 🚀
        title = title_1 + '   ' + title_2
        plt.title(title, fontsize=9.5, loc='left', color='#222222')

        plt.figtext(
            0.0835,  # x 坐标（1.0 是右侧边缘，0.95 是稍微靠左一点）
            0.95, desc,  # y 坐标（0.5 是中间）
            fontsize=14,
            fontdict={'family': 'Microsoft JhengHei'},
            color='#222222',
            ha='left',  # 左对齐
            va='top',  # 垂直居中
            # bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')  # 添加背景框
        )

        # LOGO  ------------------------------------------------------------------------------------------------------

        outline_width = 4  # 描边宽度
        text_color = 'blue'  # 文字颜色
        font_size = 9  # 字体大小
        font_weight = 'bold'  # 字体加粗

        light_blue = (0.3, 0.507, 0.932, 1)  # RGB 值，对应 #ADD8E6
        outline_color = (0.6, 0.6, 0.7, 1)  # RGB 值，对应 #00008B

        # 绘制描边效果（多次绘制文字）
        for dx in [-outline_width, 0, outline_width]:
            plt.text(
                0.93 + detail_extra_len / 5.1, -0.1, 'Multi',
                fontsize=font_size,
                fontweight=font_weight,
                fontdict={'family': 'Calibri'},
                color=outline_color,  # 描边颜色
                transform=plt.gca().transAxes,  # 使用相对坐标
                ha='left', va='bottom',  # 对齐方式
                zorder=2  # 确保描边在文字下方

            )

        # 绘制实际文字
        plt.text(
            0.93 + detail_extra_len / 5.1, -0.1, 'Multi-DPLS',
            fontsize=font_size,
            fontweight=font_weight,
            fontdict={'family': 'Calibri'},
            color=light_blue,  # 文字颜色
            transform=plt.gca().transAxes,  # 使用相对坐标
            ha='left', va='bottom',  # 对齐方式
            zorder=1,  # 确保文字在描边上方
            bbox=dict(  # 添加描边
                edgecolor='none',  # 描边颜色
                facecolor='none',  # 背景颜色（无）
                linewidth=1)  # 描边宽度
        )

        # Diagonal  --------------------------------------------------------------------------------------------------

        plt.plot([0, 1], [1, 0], color='black', linestyle='--', label='Diagonal', linewidth=0.3)

        # Details  ---------------------------------------------------------------------------------------------------

        if details:
            params_text = (
                f'-Seed-\n'
                f"func_seeds = {self.func_seed}\n"
                f"x_seeds = {self.x_seed}\n"
                f"test_seed = {self.test_seed}\n\n"

                f'-Size-\n'
                f"size_of_sample = {self.train_samples}\n"
                f"size_of_test = {self.test_samples}\n\n"

                f'-Function-\n'
                f"use_x_num = {self.use_x_num}\n"
                f"x_num = {self.x_num}\n"
                f"x_to_x_num = {self.x_to_x_num}\n"
                f"redundancy = {self.keep_irrelevant}\n\n"

                f'-P-\n'
                f"cv = {self.cv}\n"
                f"fit_mode = {self.fit_mode}\n\n"

                f'-PLS-\n'
                f"tolerance = {self.tol}\n"
                f"max_iter = {self.max_iter}\n"
                f"square={self.square}\n"
                f"whiten={self.whiten}\n"
                f"eig_solver={self.eig_solver}\n\n"

                f'-Plot-\n'
                f"region = {self.R_threshold}\n"
                f"dots_num = {self.dots_num}\n\n"

            )

            if notice:
                import textwrap
                wrapped_text = textwrap.fill(notice, width=36)
                params_text += f"-Notice-\n{wrapped_text}"

            plt.figtext(
                0.975,  # x 坐标（1.0 是右侧边缘，0.95 是稍微靠左一点）
                0.865,  # y 坐标（0.5 是中间）
                params_text,
                fontsize=7.5,
                fontweight='bold',
                color='darkgray',
                ha='right',  # 右对齐
                va='top', wrap=True,
                fontdict={'family': 'Microsoft JhengHei'},
                # 垂直居中
                # bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')  # 添加背景框
            )

        # Show ---------------------------------------------------------------------------------------------------------
        plt.tight_layout()  # 调整布局，防止文字被裁剪

        return fig

    def plot(self, x_df: pd.DataFrame, y_df: pd.DataFrame, x_label: str = '', y_label: str = '',
             desc: str = '', plot_mode: Literal['each', 'entire'] = 'entire',
             details: bool = False, notice=None):

        colors = ['#f02323', '#228855', '#334fe5', '#EA7300', '#3E3232', '#9BEC00', '#FFEB00'
                                                                                    '']
        if plot_mode == 'each':

            for c, func in enumerate(y_df.columns):
                self._plot(desc=desc, details=details, x_df=x_df[[func]], y_df=y_df[[func]],
                           x_label=x_label, y_label=y_label, color=[colors[c]], notice=notice)

        elif plot_mode == 'entire':

            fig = self._plot(desc=desc, details=details, x_df=x_df, y_df=y_df,
                       x_label=x_label, y_label=y_label, color=colors, notice=notice)
            return fig

        else:

            raise AttributeError('plot_mode must be "entire" or "each"')

    def _check_pattern(self, pattern: list, region):

        param_dict = self.__dict__
        param_dict['distance_pattern'] = pattern
        # param_dict['check_single'] = True

        pattern_R_obj = DPLS_Checker(**param_dict).check_x_equity(region=region, plot=False)
        pattern_R_true_DF = pattern_R_obj.R_true_DF
        pattern_R_return_DF = pattern_R_obj.R_return_DF

        return pattern_R_true_DF, pattern_R_return_DF

    def check_pattern(self, patterns: tuple,
                      region: tuple = (1, 0),
                      plot=True,
                      plot_mode: Literal['each', 'entire'] = 'entire',
                      desc='',
                      notice='',
                      details=False,
                      **kwargs, ):

        pattern_returns_list = []
        pattern_trues_list = []

        for pattern in patterns:
            pattern_true, pattern_return = self._check_pattern(pattern=pattern, region=region)

            pattern_col_name = [f'{str(pattern)}' + str(column_i) for column_i in pattern_return.columns]
            pattern_return.columns = pattern_col_name
            pattern_true.columns = pattern_col_name

            pattern_returns_list.append(pattern_return)
            pattern_trues_list.append(pattern_true)

        pattern_returns = pd.concat(pattern_returns_list, axis=1)
        pattern_trues = pd.concat(pattern_trues_list, axis=1)

        if plot:
            self.plot(desc=desc, notice=notice, plot_mode=plot_mode,
                      x_df=1 - pattern_trues, y_df=pattern_returns, details=details,
                      x_label='DPLS_R^2', y_label='Noise (1 - R<y_exp, y_obs>^2)')

        return pattern_trues, pattern_returns


if __name__ == '__main__':
    result_Df = pd.DataFrame()

    xx_Euc = DPLS_Checker(
                          thread=1,
                          max_iter=50,
                          func_seed=[0,5,14],
                          x_seed=[5,10,14],

                          train_samples=500,
                          use_x_num=[1,2,3],
                          x_num=[1,0,0],
                          x_to_x_num=[0,1,1],
                          x_to_x_level=3,
                          use_x_func=["正弦函数"],
                          use_xtox_func=["积函数"],

                          cv=5,

                          keep_irrelevant=False,
                          redundancy=False,
                          square=False,
                          transpose=True,
                          fit_mode="Fit_rectify",
                          whiten=False,
                          tol=1e-5,
                          eig_solver='sklearn',

                          x_start=-1, x_end=1,
                          distance_pattern=['Euc'],
                          dots_num=20,
                          )

    print(xx_Euc.func_names)
    xx_Euc.check_x_equity(region=(1, 0),
                          plot=True,
                          desc=r'DPLS8月02复现,CV Top5>(p/n**0.6*m), Max([fit pred R2 -AVERAGE (np logistic)])',
                          # 'DPLS多阶互作, Max([fit_pred_R2 - AVERAGE_(np_logistic)]), max_iter_60, 实数'
                          details=True, plot_mode='entire',
                          notice='δ_fit=fit_pred_R2-fit_pred_R2_\nAVERAGE_(np_logistic)\n=1/(1+np.exp(-a*(ln(p/n)-b)))'
                          )

    # xx_Euc.R_return_DF.to_excel('x2x3_Euc.xlsx')

    # Euc_R = checker.check_x_equity().X_R
    #
    # pattern_divide = pd.DataFrame(pattern_returns.iloc[:, 0] / pattern_returns.iloc[:, 1], columns=['Euc/Mah - x_1*x_2'])
    # pattern_trues['Euc/Mah - x_1*x_2'] = pattern_trues.iloc[:, 0]
    # checker.plot(x_df=1-pattern_trues, y_df=pattern_divide)
