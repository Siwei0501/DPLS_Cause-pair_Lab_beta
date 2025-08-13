from typing import Literal, Union, Callable, Iterable

import numpy as np
import pandas as pd
from causallearn.utils.cit import CIT
from hyppo.independence import Hsic
from joblib import Parallel, delayed
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from tqdm import tqdm

from GUI_functions import test_tools_v312
from GUI_functions.DPLS_jj import DPLS

Pre_process_Option = Literal[
    'Xmean', 'None', 'add_noise', 'drop_duplicates_mean', 'to_DPLS_pred', 'subsidiary_sampling', 'normalize', 'regionalitze']
Pre_process_Iterable = Union[Pre_process_Option, Iterable[Pre_process_Option]]

Method_Option = Literal[
    'DPLSR', 'DPLS', 'P_DPLSR', 'MIDC', 'DPLSe_KCI', 'DPLSe_P_KCI', 'DPLSe_HSIC',
    'PATH', 'Sum', 'PersonR', 'is_Linear', 'GS', '','CV', "Sort_by_reason",
    'DPLS_predR', 'break_DPLSR', 'CMVe_DPLSR', 'aCMV_DPLSe_KCI', 'chain_stability', 'CMVe_DPLSe_KCI', 'PATH_tender', 'shuffle_path']


def find_inflection_points(x, y):
    # 一阶差分（近似导数）
    dy = np.diff(y) / np.diff(x)

    # 二阶差分（近似二阶导）
    ddy = np.diff(dy) / np.diff(x[:-1])  # x[:-1] 是因为 dy 的长度比 x 少1

    # 找出二阶导数绝对值最大的点（即拐点）
    turning_index = np.argmax(np.abs(ddy)) + 1  # +1 是因为差分导致索引偏移

    # 拐点坐标
    turning_point = (x[turning_index], y[turning_index])
    return turning_point[0]


def find_inflection_points_window(x, y, window_size=2):
    # 拟合一个三次样条，设置平滑度s=0保证通过所有点
    spline = UnivariateSpline(x, y, k=3, s=0)

    # 计算一阶导数和二阶导数
    dy = spline.derivative(n=1)(x)
    ddy = spline.derivative(n=2)(x)

    # 寻找二阶导数为零处（即拐点），可通过过零点检测
    signs = np.sign(ddy)
    zero_crossings = np.where(np.diff(signs))[0]

    # 拐点位置
    turning_point_x = x[zero_crossings[0]]
    turning_point_y = y[zero_crossings[0]]

    return turning_point_x, turning_point_y


def estimate_with_matern(x, y):
    # 确保x是二维数组 (n_samples, n_features)
    X = x.reshape(-1, 1)

    # 定义Matern核函数 (常用ν=1.5或2.5)
    kernel = Matern(length_scale=1.0, nu=1.5)

    # 创建高斯过程回归模型
    gp = GaussianProcessRegressor(kernel=kernel)

    # 拟合模型
    gp.fit(X, y)

    # 预测x=0处的值
    X_pred = np.array([[0.0]])
    y_pred, sigma = gp.predict(X_pred, return_std=True)

    return y_pred[0]  # 返回估计值


def col_mean(data: pd.DataFrame, host_col, guest_col):
    host = data.drop_duplicates(subset=host_col)[[host_col]]
    guest = []

    for host_i in host[host_col]:
        guest_i = data[data[host_col] == host_i][[guest_col]]
        guest.append(np.mean(guest_i))

    host[guest_col] = guest

    return host


def cal_DPLSR(file_value:pd.DataFrame, **kwargs):

    value_copy = file_value.copy()

    DPLS_obj = DPLS(**kwargs).fit(value_copy[[kwargs['reason']]], value_copy[[kwargs['result']]], **kwargs)

    return DPLS_obj.R2[0], DPLS_obj.p[0]


def cal_DPLS_obj(file_value:pd.DataFrame, **kwargs):

    value_copy = file_value.copy()

    DPLS_needed_param = ["_max_iter_", "_R2_", "_cv_R2_", "_fit_R2_", "_p_", "_cv_p_", "_fit_p_", "_y_pred_R2_", ]
    piked_DPLS_params = []


    for param in DPLS_needed_param:

        if kwargs.get(param, False):

            piked_DPLS_params.append(param[1:-1])

    if not piked_DPLS_params:

        piked_DPLS_params = ["R2"]


    DPLS_obj = DPLS(**kwargs).fit(value_copy[[kwargs['reason']]], value_copy[[kwargs['result']]], **kwargs)

    needed_param = []

    for param in piked_DPLS_params:

        if param == 'y_pred_R2':

            needed_param.extend(DPLS_obj.__dict__['y_pred_R2'][0])

        elif param == 'max_iter':

            needed_param.append(DPLS_obj.__dict__['max_iter'])

        else:
            needed_param.extend(DPLS_obj.__dict__[param])


    return needed_param


def cal_PersonR(file_value:pd.DataFrame,R2=True,  **kwargs):
    value_copy = file_value.copy()

    R = test_tools_v312.calculate_corr(X=value_copy[[kwargs['reason']]], y=value_copy[[kwargs['result']]])
    return R

def cal_CV(file_value: pd.DataFrame, **kwargs):
    value_copy = file_value.copy()
    CV = value_copy[[kwargs['reason']]].std() / value_copy[[kwargs['reason']]].mean()
    return CV

def cal_chain_stability(file_value: pd.DataFrame, Chain_mode:Literal['flow', 'tree']='flow', Chain_len:int=3,  **kwargs):

    value_copy = file_value.copy()
    value_copy = value_copy.dropna(axis=0, how='any')

    a = value_copy[[kwargs['reason']]]
    b = value_copy[[kwargs['result']]]

    if Chain_mode == 'flow':

        Chain_R = []

        a_pred=a
        b_pred=b

        for c in range(Chain_len):

            chain_obj = DPLS(**kwargs).fit(a_pred,b_pred, **kwargs)

            chain_R_c = chain_obj.R2[0]

            Chain_R.append(chain_R_c)

            b_pred = chain_obj.y_pred[0]

            a_obj = DPLS(**kwargs).fit(b_pred, a_pred, **kwargs)
            a_pred = a_obj.y_pred[0]

            if np.any(np.isnan(chain_obj.R2[0])) or np.any(np.isnan(a_obj.R2[0])):
                return 0


    elif Chain_mode == 'tree':

        Chain_R = []

        a_pred=a

        for c in range(Chain_len):

            chain_obj = DPLS(**kwargs).fit(a_pred,b, **kwargs)

            chain_R_c = chain_obj.R2[0]
            Chain_R.append(chain_R_c)

            b_pred = chain_obj.y_pred[0]

            a_obj = DPLS(**kwargs).fit(b_pred, a, **kwargs)
            a_pred = a_obj.y_pred[0]

            if np.any(np.isnan(chain_obj.R2[0])) or np.any(np.isnan(a_obj.R2[0])):
                return 0


    else:
        Chain_R = cal_chain_stability(file_value, Chain_mode='flow', Chain_len=Chain_len, **kwargs)


    Chain_R = np.array(Chain_R)

    return np.std(Chain_R)


def cal_CMV(file_value: pd.DataFrame, CMV_mode:Literal['distance', 'origin']='distance', CMV_num:int=1, **kwargs):

    from sklearn.decomposition import PCA
    from GUI_functions.DPLS_Direct import DPLS_distance

    value_copy = test_tools_v312.stdize(file_value.copy())
    value_copy = value_copy.dropna(axis=0, how='any')

    if CMV_mode == 'origin':
        CMV_num = 1

    if CMV_num >file_value.shape[0]:
        CMV_num = file_value.shape[0]

    if CMV_mode == 'distance':

        cmv_distance_matrix = DPLS_distance(np.array(value_copy), distance_pattern=['Euc'])
        cmv_obj = PCA(n_components=CMV_num).fit(cmv_distance_matrix)
        cmv = test_tools_v312.to_2D_ary(cmv_obj.components_).T

    elif CMV_mode == 'origin':

        cmv_obj = PCA(n_components=CMV_num).fit(np.array(value_copy).T)
        cmv = test_tools_v312.to_2D_ary(cmv_obj.components_).T

    else:
        cmv = cal_CMV(file_value, CMV_mode='distance', **kwargs)

    return cmv


def cal_CMVe(file_value: pd.DataFrame, **kwargs):

    value_copy = file_value.copy()
    value_copy = value_copy.dropna(axis=0, how='any')

    a = value_copy[[kwargs['reason']]]
    b = value_copy[[kwargs['result']]]

    cmv = cal_CMV(file_value, **kwargs)

    b_obj = DPLS(**kwargs).fit(cmv, b, **kwargs)
    b_pred = b_obj.y_pred[0]
    b_e = b-b_pred

    a_obj = DPLS(**kwargs).fit(cmv, a, **kwargs)
    a_pred = a_obj.y_pred[0]
    a_e = a-a_pred

    return a_e, b_e


def cal_CMVe_DPLSR(file_value: pd.DataFrame,  **kwargs):

    a_e, b_e = cal_CMVe(file_value, **kwargs)

    cmv_R = DPLS(**kwargs).fit(a_e, b_e, **kwargs).R2[0]

    return cmv_R

def cal_CMVe_DPLSe_KCI(file_value: pd.DataFrame, **kwargs):

    a_e, b_e = cal_CMVe(file_value, **kwargs)
    DPLSe_KCI_obj = DPLS(**kwargs).fit(a_e, b_e, **kwargs)
    b_e_DPLSe_KCI = cal_X_e_KCI(DPLSe_KCI_obj)

    return b_e_DPLSe_KCI

def cal_aCMV_DPLSe_KCI(file_value: pd.DataFrame, **kwargs):

    value_copy = test_tools_v312.stdize(file_value.copy())

    a = value_copy[[kwargs['reason']]]
    b = value_copy[[kwargs['result']]]

    if 'CMV_mode' in kwargs:
        del kwargs['CMV_mode']

    cmv = cal_CMV(file_value, CMV_mode='distance', **kwargs)

    a_cmv = np.hstack((np.array(a), cmv))

    DPLSe_KCI_obj = DPLS(**kwargs).fit(a_cmv, b, **kwargs)
    b_DPLSe_KCI = cal_X_e_KCI(DPLSe_KCI_obj)

    return b_DPLSe_KCI


def cal_X_e_HSIC(file_value: pd.DataFrame,  **kwargs):

    # if P is not None:
    #     y_pred = DPLS_obj.y_preds[0][:, P]
    # else:
    #     y_pred = DPLS_obj.y_pred[0][:, P]

    value_copy = file_value.copy()
    value_copy = test_tools_v312.stdize(value_copy)

    reason_result_obj = DPLS(**kwargs).fit(value_copy[kwargs['reason']], value_copy[kwargs['result']], **kwargs)



    e = np.array(reason_result_obj.y - reason_result_obj.y_pred[0])

    hsic = Hsic()
    stat, p_value =  hsic.test(x=np.array(reason_result_obj.X), y=e)

    return p_value


def cal_X_e_P_HSIC(file_value: pd.DataFrame, P_mode: Literal['min', 'max', 'mean'] = 'mean', **kwargs):

    value_copy = file_value.copy()

    reason_result_obj = DPLS(**kwargs).fit(value_copy[kwargs['reason']], value_copy[kwargs['result']], **kwargs)
    result_reason_obj = DPLS(**kwargs).fit(value_copy[kwargs['result']], value_copy[kwargs['reason']], **kwargs)

    if np.any(np.isnan(result_reason_obj.R2[0])) or np.any(np.isnan(result_reason_obj.R2[0])):

        return 0, 0, 0, 0

    reason_result_P = reason_result_obj.p[0]
    result_reason_P = result_reason_obj.p[0]

    if P_mode == 'min':

        P = min(reason_result_P, result_reason_P)

    elif P_mode == 'max':

        P = max(reason_result_P, result_reason_P)

    elif P_mode == 'mean':

        P = int(np.mean([reason_result_P, result_reason_P]))

    else:
        raise ValueError('P_mode must be one of "min", "max", "mean"')


    reason_result_HSIC = cal_X_e_HSIC(reason_result_obj, **kwargs)
    result_reason_HSIC = cal_X_e_HSIC(result_reason_obj, **kwargs)

    return reason_result_HSIC, result_reason_HSIC

# 求 DPLS 预测值的残差 e 与 x 的 KCI 值
def cal_X_e_KCI(DPLS_obj: DPLS, P: None | int = None, **kwargs):


    if np.isnan(DPLS_obj.R2[0]):
        return 0,0

    if P is not None:
        y_pred = DPLS_obj.y_preds[0][:, P]
        y_pred = y_pred.reshape(-1, 1)
    else:
        y_pred = DPLS_obj.y_pred[0]

    e = np.array(DPLS_obj.y - y_pred).reshape(-1, 1)

    X_e = np.hstack([np.array(DPLS_obj.X), e])
    X_shape1 = DPLS_obj.X.shape[1]
    X_e_KCI_obj = CIT(X_e, method='kci')
    X_e_KCI_value = X_e_KCI_obj(1, range(X_shape1))

    return X_e_KCI_value  # KCI(y, x, cmv)

def cal_Xe_KCI(file_value: pd.DataFrame, **kwargs):
    value_copy = file_value.copy()

    reason = value_copy[[kwargs['reason']]]
    result = value_copy[[kwargs['result']]]

    reason_result_obj = DPLS(**kwargs).fit(reason, result, **kwargs)
    DPLSe_KCI = cal_X_e_KCI(reason_result_obj)
    return DPLSe_KCI


def cal_X_e_P_KCI(file_value: pd.DataFrame, P_mode: Literal['min', 'max', 'mean'] = 'mean', **kwargs):
    value_copy = file_value.copy()

    reason_result_obj = DPLS(**kwargs).fit(value_copy[kwargs['reason']], value_copy[kwargs['result']], **kwargs)
    result_reason_obj = DPLS(**kwargs).fit(value_copy[kwargs['result']], value_copy[kwargs['reason']], **kwargs)

    if np.any(np.isnan(result_reason_obj.R2[0])) or np.any(np.isnan(result_reason_obj.R2[0])):

        return 0,0,0,0

    reason_result_P = reason_result_obj.p[0]
    result_reason_P = result_reason_obj.p[0]

    if P_mode == 'min':

        P = min(reason_result_P, result_reason_P)

    elif P_mode == 'max':

        P = max(reason_result_P, result_reason_P)

    elif P_mode == 'mean':

        P = int(np.mean([reason_result_P, result_reason_P]))

    else:
        raise ValueError('P_mode must be one of "min", "max", "mean"')

    reason_result_KCI = cal_X_e_KCI(reason_result_obj, P, **kwargs)
    result_reason_KCI = cal_X_e_KCI(result_reason_obj, P, **kwargs)

    return reason_result_KCI[0], reason_result_KCI[1], result_reason_KCI[0], result_reason_KCI[1]


def cal_P_DPLSR(file_value: pd.DataFrame, P_mode: Literal['min', 'max', 'mean'] = 'mean', **kwargs):
    value_copy = file_value.copy()

    reason_result_obj = DPLS(**kwargs).fit(value_copy[kwargs['reason']], value_copy[kwargs['result']], **kwargs)
    result_reason_obj = DPLS(**kwargs).fit(value_copy[kwargs['result']], value_copy[kwargs['reason']], **kwargs)

    if np.any(np.isnan(result_reason_obj.p[0])) or np.any(np.isnan(result_reason_obj.p[0])):
        return 0,0

    reason_result_P = reason_result_obj.p[0]
    result_reason_P = result_reason_obj.p[0]

    if P_mode == 'min':

        P = min(reason_result_P, result_reason_P)

    elif P_mode == 'max':

        P = max(reason_result_P, result_reason_P)

    elif P_mode == 'mean':

        P = int(np.mean([reason_result_P, result_reason_P]))

    else:
        raise ValueError('P_mode must be one of "min", "max", "mean"')

    return reason_result_obj.y_pred_R2[0][P], result_reason_obj.y_pred_R2[0][P]


def cal_DPLS_pred_R(file_value, **kwargs):

    value_copy = file_value.copy()
    X = value_copy[kwargs['reason']]
    y = value_copy[kwargs['result']]
    obj = DPLS(**kwargs).fit(X.astype(float), y.astype(float), **kwargs)
    y_pred = obj.y_pred[0]
    y_pred_R = test_tools_v312.calculate_corr(X, y_pred)[0]

    return y_pred_R


def cal_break_DPLSR(file_value: pd.DataFrame, break_parts=2, **kwargs):

    if break_parts < 2:
        break_parts = 2

    value_copy = file_value.copy()
    value_sorted = value_copy.sort_values(by=kwargs['reason'], ascending=False).reset_index(drop=True)

    # 删除异常值
    mean = value_sorted.mean()
    std = value_sorted.std()
    mask = (value_sorted >= (mean - 3 * std)) & (value_sorted <= (mean + 3 * std))
    value_sorted = value_sorted[mask.all(axis=1)].reset_index(drop=True)
    reason_max = value_sorted[kwargs['reason']].max()
    reason_min = value_sorted[kwargs['reason']].min()
    # 求断点
    break_point = np.abs(reason_max - reason_min) / break_parts
    idx_R2 = []

    for part_i in range(break_parts):

        idx_i = value_sorted[(value_sorted[kwargs['reason']] > reason_min + break_point*(part_i)) & (value_sorted[kwargs['reason']] <= reason_min + break_point*(part_i+1))].index

        idx_obj = DPLS(**kwargs).fit(value_sorted.loc[idx_i, kwargs['reason']], value_sorted.loc[idx_i, kwargs['result']], **kwargs)
        idx_R2.append(idx_obj.R2[0])

    idx_R2 = np.array(idx_R2)
    interval_sum = np.sum(idx_R2[:-1] - idx_R2[1:])

    return interval_sum

# 特定分布的间隔取 x 及其 y 均值
def cal_meanX(file_value: pd.DataFrame, mean_ratio=0.2, Xmean_mode: Literal['uniform', 'normal'] = 'normal', Xmean_window=2,
              dropna=True, **kwargs):
    data_copy = file_value.copy()

    X = file_value[kwargs['reason']]
    y = file_value[kwargs['result']]

    X = test_tools_v312.to_2D_ary(X)
    y = test_tools_v312.to_2D_ary(y)
    num = int(X.shape[0] * mean_ratio)

    if (X.shape[1] or y.shape[1]) > 1:
        raise 'cal_meanX Except vector only'

    X_max = np.max(X)
    X_min = np.min(X)

    if Xmean_mode == 'uniform':

        Q = np.arange(X_min, X_max, (X_max - X_min) / num).tolist()
        Q.append(X_max)

    elif Xmean_mode == 'normal':

        X_mean = np.mean(X)
        X_std = np.std(X)

        F_max = norm.cdf(X_max, loc=X_mean, scale=X_std)
        F_min = norm.cdf(X_min, loc=X_mean, scale=X_std)

        probabilities = [F_min + i * (F_max - F_min) / num for i in range(1, num)]

        Q = [norm.ppf(p, loc=X_mean, scale=X_std) for p in probabilities]
        Q.insert(0, X_min)
        Q.append(X_max)

    else:
        raise AttributeError(f'不存在的Xmean_mode: {Xmean_mode}')

    container_DF = pd.DataFrame(columns=[kwargs['reason'], kwargs['result']])

    for i in range(len(Q) - Xmean_window):

        A = X[(X >= Q[i]) & (X <= Q[i + Xmean_window])]
        B = y[(X >= Q[i]) & (X <= Q[i + Xmean_window])]

        if A.shape[0] > 0:
            container_DF.loc[i] = [np.mean(A), np.mean(B)]
        else:
            container_DF.loc[i] = [np.nan, np.nan]

    if dropna:
        container_DF.dropna(subset=kwargs['reason'], inplace=True)

    return container_DF


def cal_path(file_value: pd.DataFrame, Path_centre:bool=False, Path_normal:bool=False,  Path_window:int=1, Path_abs=True, Sort_by:Literal['reason', 'all']='all',  **kwargs):

    value_copy = file_value.copy()
    value_copy = value_copy.dropna(axis=0, how='any')

    if Sort_by == 'reason':
        value_sort = value_copy.sort_values(by=[kwargs['reason']])
    elif Sort_by == 'all':
        value_sort = value_copy.sort_values(by=[kwargs['reason'], kwargs['result']])
    else:
        value_sort = value_copy.sort_values(by=[kwargs['reason'], kwargs['result']])

    reason = np.array(value_sort[kwargs['reason']])
    result = np.array(value_sort[kwargs['result']])

    if Path_centre:

        reason = reason - np.mean(reason)
        result = result - np.mean(result)

    window_compensation = [0]*Path_window

    reason_ = np.append(reason[Path_window:], window_compensation)
    result_ = np.append(result[Path_window:], window_compensation)


    if Path_abs:

        reason_interval = np.abs(reason - reason_)
        result_interval = np.abs(result - result_)

    else:

        reason_interval = reason - reason_
        result_interval = result - result_

    reason_length = np.sum(reason_interval)
    result_length = np.sum(result_interval)

    try:

        reason_value = reason_length / value_copy.shape[0]
        result_value = result_length / value_copy.shape[0]


        if Path_normal:
            reason_value = reason_value / np.std(reason)
            result_value = result_value / np.std(result)

    except ZeroDivisionError:

        print("Path 过程检测到样本数=0, 已返回空值")

        return np.nan, np.nan

    return reason_value, result_value



def cal_shuffle_path(file_value: pd.DataFrame, Path_centre:bool=False, Path_normal:bool=False,  Path_window:int=1, Path_abs=True, Shuffle_times=1000,  **kwargs):


    value_copy = file_value.copy()

    reason_list = []
    result_list = []

    for i in range(Shuffle_times):

        value_shuffled = value_copy.sample(frac=1, random_state=i).reset_index(drop=True)
        reason_value, result_value = cal_path(file_value=value_shuffled, Path_centre=Path_centre, Path_normal=Path_normal, Path_window=Path_window, **kwargs)
        reason_list.append(reason_value)
        result_list.append(result_value)

    return np.mean(np.array(reason_list)), np.mean(np.array(result_list))




def cal_path_tender(file_value: pd.DataFrame, Tender_length:int=5, Path_centre:bool=False, Path_normal:bool=False, **kwargs):

    value_copy = file_value.copy()

    if len(value_copy) < Tender_length:

        return 0, 0

    reason_tender_list = []
    result_tender_list = []

    for t in range(Tender_length):

        t += 1

        if 'Path_window' in kwargs:

            del kwargs['Path_window']

        reason_tender_i, result_tender_i = cal_path(value_copy, Path_centre, Path_normal, Path_window=t, **kwargs)
        reason_tender_list.append(reason_tender_i)
        result_tender_list.append(result_tender_i)

    return result_tender_list



def cal_sum(file_value: pd.DataFrame, **kwargs):
    value_copy = file_value.copy()

    reason = value_copy[kwargs['reason']].sum()

    return reason

def cal_std(file_value: pd.DataFrame, **kwargs):
    value_copy = file_value.copy()

    reason = value_copy[kwargs['reason']].std()

    return reason

def cal_mean(file_value: pd.DataFrame, **kwargs):

    value_copy = file_value.copy()
    reason = value_copy[kwargs['reason']].mean()

    return reason


def cal_GS(file_value: pd.DataFrame, GS_ratio=0.6, GS_window=2, GS_core='Matern', **kwargs):
    if GS_ratio > 1:
        raise 'cal_GS(GS_ratio) must be <= 1'

    # copy
    value_copy = test_tools_v312.normalize(file_value.copy())
    value_copy = (value_copy.sort_values(by=[kwargs['reason']]))

    reason = np.array((value_copy[[kwargs['reason']]]))
    result = np.array((value_copy[[kwargs['result']]]))

    # distance
    reason_distance = np.abs(reason - reason.T)
    result_distance = np.abs(result - result.T)

    # 确定间隔
    h_max = (np.max(reason) - np.min(reason)) / 2
    pairs_in_h_max = np.sum(reason_distance <= h_max)
    h_interval = h_max / (file_value.shape[0] * GS_ratio)

    clusters_basic = np.append(np.arange(0, h_max, h_interval), h_max)
    clusters_center = (clusters_basic[:-1] + clusters_basic[1:]) / 2

    clusters = np.array([clusters_basic[:-1], clusters_center]).T.flatten()
    clusters = np.append(clusters, h_max)

    GS_Dict = {}
    GS_pair_Dict = {}
    for h in range(len(clusters) - GS_window):
        h_low = clusters[h]
        h_high = clusters[h + GS_window]

        pairs_in_h = np.logical_and(reason_distance >= h_low, reason_distance <= h_high)
        distances_in_h = result_distance * pairs_in_h

        s_of_h = np.sum(distances_in_h ** 2)
        CV_of_h = s_of_h / np.sum(pairs_in_h)

        GS_Dict[h] = CV_of_h
        GS_pair_Dict[h] = np.sum(pairs_in_h)

    GS_Dataframe = pd.DataFrame.from_dict(GS_Dict, orient='index', columns=['GS'])
    GS_Dataframe = GS_Dataframe.dropna()

    a, CV_a = find_inflection_points_window(GS_Dataframe.index.to_numpy(), GS_Dataframe['GS'].to_numpy(),
                                            window_size=GS_window)

    if GS_core == 'Matern':
        CV_0 = estimate_with_matern(GS_Dataframe.index.to_numpy(), GS_Dataframe['GS'].to_numpy())
    else:
        raise f'Gs_core must in [Matern]'

    return CV_0, CV_a, a


def is_Linear(file_value: pd.DataFrame,Linear_times:int=5, Linear_ratio:float=0.7, **kwargs):
    value_copy = file_value.copy()
    reason_list = []

    for i in range(Linear_times):

        value_sample = value_copy.sample(int(value_copy.shape[0] * Linear_ratio), random_state=i)

        reason_result_predR = cal_DPLS_pred_R(value_sample, **kwargs)
        reason_list.append(reason_result_predR)


    return np.mean(np.array(reason_list))

    # coef = np.mean(np.array(reason_list)) / np.mean(np.array(result_list))

    # if coef > 1.2 or coef < 0.833:
    #     return 0
    # else: 
    #     return 1


def MIDC(X, y, parts=3, **kwargs):
    #

    X = test_tools_v312.to_2D_ary(X)
    y = test_tools_v312.to_2D_ary(y)

    MIDC_result = []
    each = X.shape[0] // parts

    for part in range(parts):

        if part == parts - 1:
            X_i = X[each * part:, :]
            y_i = y[each * part:, :]
        else:
            X_i = X[each * part:each * (part + 1), :]
            y_i = y[each * part:each * (part + 1), :]
        R_i = DPLS().fit(X_i, y_i).R2[0]
        MIDC_result.append(R_i)

    return MIDC_result


def to_DPLS_y_pred(file_value: pd.DataFrame, **kwargs):
    data_copy = file_value.copy()

    reason = data_copy[kwargs['reason']]
    result = data_copy[kwargs['result']]

    reason = test_tools_v312.to_2D_ary(reason)
    result = test_tools_v312.to_2D_ary(result)

    result_obj = DPLS(**kwargs).fit(X=reason.astype(float), y=result.astype(float), **kwargs)

    result_y_pred = result_obj.y_pred[0]

    data_copy[kwargs['result']] = result_y_pred

    return data_copy


def add_noise(file_value: pd.DataFrame, data_R=None,
              noise_level: float = 0.1,
              noise_mode: Literal['uniform', 'normal'] = 'normal', add_on: Literal['reason', 'result'] = 'result',
              adjust=False, **kwargs):
    data_copy = file_value.copy()
    data_std = data_copy[kwargs[add_on]].std()
    use_level = noise_level

    if adjust:
        if data_R is not None:
            use_level = data_R * noise_level
        else:
            use_level = DPLS(**kwargs).fit(data_copy[kwargs['reason']], data_copy[kwargs['result']], **kwargs).R2[0] * noise_level

    if noise_mode == 'uniform':
        noise = np.random.uniform(-data_std * use_level, data_std * use_level, size=data_copy.shape[0])

    elif noise_mode == 'normal':
        noise = np.random.normal(loc=0, scale=data_std * use_level, size=data_copy.shape[0])

    else:
        raise f'Unknown noise_mode: {noise_mode}'

    data_copy[kwargs[add_on]] = data_copy[[kwargs[add_on]]] + noise.reshape(-1, 1)
    return data_copy


def pair_subsidiary_sampling(file_value: pd.DataFrame, subsidiary_width=2, **kwargs):
    data_copy = file_value.copy()
    data_copy = data_copy.sort_values(by=kwargs['reason'])
    data_copy = data_copy.groupby(kwargs['reason'], as_index=False).agg({kwargs['result']: 'mean'})

    # # 随机打乱行 - 使用sample方法
    # # shuffled_df = df.sample(frac=1).reset_index(drop=True)
    data_copy = data_copy.sort_values(by=kwargs['reason'])

    group_key = np.arange(len(data_copy)) // subsidiary_width

    # 对每组随机取一行
    if 'seed' in kwargs.keys():
        seed = kwargs['seed']
    else:
        seed = None

    try:

        data_pair = data_copy.groupby(group_key).apply(
            lambda x: x.sample(1, random_state=seed)).reset_index().set_index(['level_1'], drop=True)
        data_pair.drop('level_0', axis=1, inplace=True)

    except KeyError:
        print("pair_subsidiary_sampling 过程失败,已返回原始值")
        return file_value

    return data_pair



def subsidiary_sampling(file_value: pd.DataFrame, subsidiary_samples: int, seed=None, **kwargs):
    data_copy = file_value.copy()
    data_copy = data_copy.sample(n=subsidiary_samples, random_state=seed)

    return data_copy


def drop_duplicates_mean(file_value: pd.DataFrame, **kwargs):
    data_copy = file_value.copy()
    data_copy = data_copy.groupby(kwargs['reason'])[kwargs['result']].mean().reset_index()
    return data_copy

def Sort_by_reason(file_value: pd.DataFrame, **kwargs):
    data_copy = file_value.copy()
    data_copy = data_copy.sort_values(by=kwargs['reason'])
    return data_copy

def drop_NA(file_value: pd.DataFrame, **kwargs):
    data_copy = file_value.copy()
    data_copy = data_copy.dropna(axis=0, how='any')
    return data_copy

def drop_Bias(file_value: pd.DataFrame, Bias_threshold:int=3, **kwargs):
    data_copy = file_value.copy()
    """
        删除 DataFrame 中任意一列存在偏倚值（超出均值 ± n倍标准差）的行。

        参数:
        - df: pandas DataFrame，输入数据。
        - n: int 或 float，标准差倍数，默认 3。

        返回:
        - pandas DataFrame，清理后的数据（不包含偏倚行）。
        """
    # 计算每列的均值 ± n倍标准差
    lower_bounds = data_copy.mean() - Bias_threshold * data_copy.std()
    upper_bounds = data_copy.mean() + Bias_threshold * data_copy.std()

    # 标记偏倚值（True表示该位置的值是偏倚值）
    outliers = (data_copy < lower_bounds) | (data_copy > upper_bounds)

    # 如果某行有任意一个偏倚值，则标记为True（需要删除）
    rows_to_drop = outliers.any(axis=1)

    # 删除这些行（保留非偏倚行）
    df_cleaned = data_copy[~rows_to_drop]

    return df_cleaned


def parallel_wrapper(
        func: Callable,
        file_value_list: list,
        desc: str,
        thread: int = 1,
        **kwargs
) -> list:

    if thread == 1:

        return_list = []
        for file_value in tqdm(file_value_list, desc=desc):
            file_return = func(file_value , **kwargs)
            return_list.append(file_return)

        return return_list

    return Parallel(n_jobs=thread)(delayed(func)(item, **kwargs) for item in tqdm(file_value_list, desc=desc))


process = {
           'normalize': test_tools_v312.stdize,
           'regionalitze': test_tools_v312.normalize,
            'drop_duplicates_mean':drop_duplicates_mean,
            'drop_NA':drop_NA,
            'drop_Bias':drop_Bias,
           'Xmean':cal_meanX,
           'add_noise':add_noise,
           'pair_subsidiary_sampling':pair_subsidiary_sampling,
           'subsidiary_sampling':pair_subsidiary_sampling,
           'to_DPLS_pred':to_DPLS_y_pred,
            "Sort_by_reason":Sort_by_reason,
           }

algorithms = {'DPLS': cal_DPLS_obj,
              'DPLSR':cal_DPLSR,
              'PATH': cal_path,
              'PersonR': cal_PersonR,
              'CV': cal_CV,
              'P_DPLSR': cal_P_DPLSR,

              'DPLSe_HSIC': cal_X_e_HSIC,
              'DPLS_predR': cal_DPLS_pred_R,
              'break_DPLSR': cal_break_DPLSR,
              'CMVe_DPLSR': cal_CMVe_DPLSR,
              'chain_stability': cal_chain_stability,
              'is_Linear': is_Linear,
              
              'shuffle_path':cal_shuffle_path,
              'GS': cal_GS,
              'PATH_tender': cal_path_tender,

              'aCMV_DPLSe_KCI': cal_aCMV_DPLSe_KCI,
              'CMVe_DPLSe_KCI': cal_CMVe_DPLSe_KCI,
              'DPLSe_KCI': cal_Xe_KCI,
              'DPLSe_P_KCI': cal_X_e_P_KCI,

              'Sum': cal_sum,
              "Std": cal_std,
              "Mean": cal_mean,

              }


def return_name(method, reverse=False, **kwargs) -> list[str]:

    if method == 'DPLS':

        DPLS_needed_param = ["_max_iter_", "_R2_", "_cv_R2_", "_fit_R2_", "_p_", "_cv_p_", "_fit_p_", "_y_pred_R2_", ]
        piked_DPLS_params = []

        for param in DPLS_needed_param:

            if kwargs.get(param, False):
                piked_DPLS_params.append(param[1:-1])

        if not piked_DPLS_params:
            piked_DPLS_params = ["R2"]

        if reverse:

            tail_char = f"(B, A)"

        else:

            tail_char = f"(A, B)"

        pre_char = f"{kwargs['pre_process']}"

        col_name = []

        for value in piked_DPLS_params:

            if value == 'y_pred_R2':

                if 'max_iter' in kwargs:

                    max_iter = kwargs['max_iter']

                else:

                    max_iter = 20

                col_name.extend([f"DPLS_{pre_char}_P{p}-R2_{tail_char}" for p in range(max_iter)])

            else:

                col_name.append(f"DPLS_{pre_char}_{value}{tail_char}")

        return col_name


    elif method == 'DPLSR':

        if reverse:
            col_name = [f"DPLS_R2{kwargs['pre_process']}(B, A)", f"P{kwargs['pre_process']}(B, A)"]
        else:
            col_name = [f"DPLS_R2{kwargs['pre_process']}(A, B)", f"P{kwargs['pre_process']}(A, B)"]


    elif method == 'P_DPLSR':

        if 'P_mode' in kwargs.keys():
            P_mode = kwargs['P_mode']
        else:
            P_mode = 'mean'

        if reverse:
            col_name = [f"BA-DPLSR-P-{P_mode}{kwargs['pre_process']}(B, A)",
                        f"BA-DPLSR-P-{P_mode}{kwargs['pre_process']}(A, B)"]
        else:
            col_name = [f"AB-DPLSR-P-{P_mode}{kwargs['pre_process']}(A, B)",
                        f"AB-DPLSR-P-{P_mode}{kwargs['pre_process']}(B, A)"]


    elif method == 'PersonR':

        if 'R2' in kwargs.keys():
            R2 = kwargs['R2']
        else:
            R2 = True

        if R2:
            R_name='R2'
        else:
            R_name='R'

        if reverse:
            col_name = [f"Person{R_name}{kwargs['pre_process']}(B, A)"]
        else:
            col_name = [f"Person{R_name}{kwargs['pre_process']}(A, B)"]

    elif method == 'MIDC':

        if reverse:
            col_name = [f"DPLS_R2{kwargs['pre_process']}(B_{i}, A_{i})" for i in range(1, kwargs['parts'] + 1)]
        else:
            col_name = [f"DPLS_R2{kwargs['pre_process']}(A_{i}, B_{i})" for i in range(1, kwargs['parts'] + 1)]

    elif method == 'DPLSe_KCI':

        if reverse:
            col_name = [f"KCI_p_{kwargs['pre_process']}(B, A_e)", f"KCI_value_{kwargs['pre_process']}(B, A_e)"]
        else:
            col_name = [f"KCI_p_{kwargs['pre_process']}(A, B_e)", f"KCI_value_{kwargs['pre_process']}(A, B_e)"]



    elif method == 'DPLSe_P_KCI':

        if 'P_mode' in kwargs.keys():
            P_mode = kwargs['P_mode']
        else:
            P_mode = 'mean'

        if reverse:
            col_name = [f"BA-DPLSe_KCI-p-{P_mode}{kwargs['pre_process']}(B, A)", f"BA-DPLSe_KCI-value-{P_mode}{kwargs['pre_process']}(B, A)",
                        f"BA-DPLSe_KCI-p-{P_mode}{kwargs['pre_process']}(A, B)", f"BA-DPLSe_KCI-value-{P_mode}{kwargs['pre_process']}(A, B)"]
        else:
            col_name = [f"AB-DPLSe_KCI-p-{P_mode}{kwargs['pre_process']}(A, B)", f"AB-DPLSe_KCI-value-{P_mode}{kwargs['pre_process']}(A, B)",
                        f"AB-DPLSe_KCI-p-{P_mode}{kwargs['pre_process']}(B, A)", f"AB-DPLSe_KCI-value-{P_mode}{kwargs['pre_process']}(B, A)",]

    elif method == 'PATH':

        if reverse:
            col_name = [f"Path_BA_{kwargs['pre_process']}(B)", f"Path_BA_{kwargs['pre_process']}(A)"]
        else:
            col_name = [f"Path_AB_{kwargs['pre_process']}(A)", f"Path_AB_{kwargs['pre_process']}(B)"]


    elif method == 'shuffle_path':

        if reverse:
            col_name = [f"Path_shuffle_BA_{kwargs['pre_process']}(B)", f"Path_shuffle_BA_{kwargs['pre_process']}(A)"]
        else:
            col_name = [f"Path_shuffle_AB_{kwargs['pre_process']}(A)", f"Path_shuffle_AB_{kwargs['pre_process']}(B)"]


    elif method == 'Sum':

        if reverse:
            col_name = [f"Sum_{kwargs['pre_process']}(B)"]
        else:
            col_name = [f"Sum_{kwargs['pre_process']}(A)"]

    elif method == "Std":

        if reverse:
            col_name = [f"STD_{kwargs['pre_process']}(B)"]
        else:
            col_name = [f"STD_{kwargs['pre_process']}(A)"]

    elif method == "Mean":

        if reverse:
            col_name = [f"Mean_{kwargs['pre_process']}(B)"]
        else:
            col_name = [f"Mean_{kwargs['pre_process']}(A)"]

    elif method == 'is_Linear':
        if reverse:
            col_name = [f"DPLS_R_{kwargs['pre_process']}(B, A^)"]
        else:
            col_name = [f"DPLS_R_{kwargs['pre_process']}(A, B^)" ]

    elif method == 'GS':

        if 'GS_core' in kwargs.keys():
            GS_core = kwargs['GS_core']
        else:
            GS_core = 'Matern'

        if reverse:
            col_name = [f"GS_0{kwargs['pre_process']}(f'{GS_core}'_A)", f"GS_a{kwargs['pre_process']}(f'{GS_core}'_A)",
                        f"a{kwargs['pre_process']}(f'{GS_core}'_A)"]
        else:
            col_name = [f"GS_0{kwargs['pre_process']}(f'{GS_core}'_B)", f"GS_a{kwargs['pre_process']}(f'{GS_core}'_B)",
                        f"a{kwargs['pre_process']}(f'{GS_core}'_B)"]

    elif method == 'DPLS_predR':

        if reverse:
            col_name = [f"DPLS_predR_{kwargs['pre_process']}(B, A)"]

        else:
            col_name = [f"DPLS_predR_{kwargs['pre_process']}(A, B)"]

    elif method == 'break_DPLSR':

        if reverse:
            col_name = [f"DPLSR_interval_{kwargs['pre_process']}(B, A)"]

        else:
            col_name = [f"DPLSR_interval_{kwargs['pre_process']}(A, B)"]

    elif method == 'DPLSe_HSIC':

        if reverse:
            col_name = [f"DPLSe_HSIC_{kwargs['pre_process']}(B, A)"]
        else:
            col_name = [f"DPLSe_HSIC_{kwargs['pre_process']}(A, B)"]


    elif method == 'CMVe_DPLSR':

        if reverse:
            col_name = [f"DPLSR_{kwargs['pre_process']}(B_e, A_e)"]
        else:
            col_name = [f"DPLSR_{kwargs['pre_process']}(A_e, B_e)"]


    elif method == 'aCMV_DPLSe_KCI':


        if reverse:
            col_name = [f"KCI_p_{kwargs['pre_process']}(B&CMV, A_e)", f"KCI_value_{kwargs['pre_process']}(B&CMV, A_e)"]
        else:
            col_name = [f"KCI_p_{kwargs['pre_process']}(A&CMV, B_e)", f"KCI_value_{kwargs['pre_process']}(A&CMV, B_e)"]

    elif method == 'CMVe_DPLSe_KCI':

        if reverse:
            col_name = [f"KCI_p_{kwargs['pre_process']}(B_e, A_ee)", f"KCI_value_{kwargs['pre_process']}(B_e, A_ee)"]
        else:
            col_name = [f"KCI_p_{kwargs['pre_process']}(A_e, B_ee)", f"KCI_value_{kwargs['pre_process']}(A_e, B_ee)"]

    elif method == 'chain_stability':

        if 'Chain_mode' in kwargs.keys():
            Chain_mode = kwargs['Chain_mode']
        else:
            Chain_mode = 'flow'

        if reverse:
            col_name = [f"STD_{Chain_mode}_{kwargs['pre_process']}(B, A)"]
        else:
            col_name = [f"STD_{Chain_mode}_{kwargs['pre_process']}(A, B)"]

    elif method == 'PATH_tender':

        if 'Tender_length' in kwargs.keys():
            Tender_length = kwargs['Tender_length']
        else:
            Tender_length = 5

        if reverse:
            col_name = [f"PATH_{Tender_length_i+1}_{kwargs['pre_process']}(B, A)" for Tender_length_i in range(Tender_length)]
        else:
            col_name = [f"PATH_{Tender_length_i+1}_{kwargs['pre_process']}(A, B)" for Tender_length_i in range(Tender_length)]

    elif method == 'CV':

        if reverse:
            col_name = [f"CV_{kwargs['pre_process']}(B, A)"]
        else:
            col_name = [f"CV_{kwargs['pre_process']}(A, B)"]

    else:
        raise f'No excepted method: {method}'

    return col_name


def return_processed_files(file_value_list: list[DataFrame], pre_process: Pre_process_Iterable = '', thread=1, **kwargs):
    used_value_list = file_value_list

    if isinstance(pre_process, str):

        if pre_process in process.keys():
            used_value_list = parallel_wrapper(func=process[pre_process], file_value_list=used_value_list,
                                               desc=pre_process, thread=thread, **kwargs)
        else:
            pass

    elif isinstance(pre_process, list):

        for process_i in pre_process:
            if process_i in process.keys():
                used_value_list = parallel_wrapper(func=process[process_i], file_value_list=used_value_list,
                                                   desc=process_i, thread=thread, **kwargs)
            else:
                pass
    else:
        raise

    return used_value_list


def return_values(file_value_list: list[DataFrame],
                  method: Method_Option = 'DPLSR', thread=1, **kwargs):

    used_value_list = file_value_list.copy()
    if method in algorithms.keys():
        values_result = parallel_wrapper(func=algorithms[method], file_value_list=used_value_list, desc=method, thread=thread, **kwargs)
    else:
        return used_value_list

    values_DF = pd.DataFrame(values_result)

    return values_DF


def return_values_DF(values_DF:DataFrame, pre_process, method, reverse=False, **kwargs):

    DF_col_name = return_name(method, reverse=reverse, pre_process=pre_process, **kwargs)
    values_DF.columns = DF_col_name

    return values_DF


def return_Cause_DF(file_value_list: list[DataFrame], pre_process: Pre_process_Iterable = '', method:Method_Option='', reverse=False, thread=1, **kwargs):

    # 默认第一列为原因
    if reverse != 1:
        reason = 0
        result = 1
    else:
        reason = 1
        result = 0

    used_value_list = file_value_list.copy()
    used_value_list = return_processed_files(used_value_list, pre_process=pre_process, thread=thread,
                                             reason=reason, result=result, **kwargs)
    return_results = return_values(file_value_list=used_value_list, method=method, thread=thread, reason=reason, result=result, **kwargs)

    return_DF = return_values_DF(return_results, pre_process=pre_process,method=method, reverse=reverse, thread=thread, **kwargs)

    return return_DF

# if __name__ == '__main__':
#
#     # Control Panel 控制面板
#
#     Relation: Literal['AB', 'BA'] = 'AB'
#
#     test_seed = 7
#     random.seed(test_seed)  # 设置随机种子
#     np.random.seed(test_seed)
#     Thread = 1  # 多线程不能调试改成 1
#
#     Threshold = [0, 1000]
#     Test_flies_num = 100
#
#
#
#     Kwargs = {'Path_centre': True, 'Path_normal':True}
#     Pre_process: Pre_process_Iterable = ['drop_duplicates_mean']
#
#     if isinstance(Pre_process, list):
#         Pre_process_name = '-'.join(Pre_process)
#     else:
#         Pre_process_name = Pre_process
#
#     if Kwargs == {}:
#         Kwargs_name: str = ''
#     else:
#         Kwargs_name: list = [f"{key}={value}" for key, value in Kwargs.items()]
#         Kwargs_name = '_{' + '-'.join(Kwargs_name) + '}'
#
#     Method: Method_Option = 'PATH'
#
# # Pre_process
#     # '': 不预处理
#     # 'add_noise': 添加噪音, noise_level: 噪音强度
#     # 'drop_duplicates_mean': 去重, 另一边求平均
#     # 'to_DPLS_pred': 转换为DPLS预测值
#     # 'subsidiary_sampling': 成对子采样
#     # 'stdize': 标准化
#     # 'normalize': 归一化
#
#
# # Methods
#     # DPLSR: DPLSR
#     # P_DPLSR: 特异P的DPLSR, P_mode: ['min', 'max', 'mean']
#     # DPLSe_KCI: e = y-y^, DPLSe_KCI=KCI(e, y)
#     # Path: delta
#     # Path-centre: delta-中心化
#     # is_Linear: 判断是否线性
#     # GS: 地统计学
#     # '': 返回预处理后的样本
#     # DPLS_predR: DPLSR(x, y^)
#
#     result_list = []
#
#     # linear_files, linear_names = select_simulation_files.select('BA', [400, 1000], test_SAMPLE=50, linear=True, seed=test_seed)
#     # non_linear_files, non_linear_names = select_simulation_files.select('BA', [400, 1000], test_SAMPLE=50, linear=False, seed=test_seed)
#     #
#     # read_files = linear_files + non_linear_files
#     # file_names = linear_names + non_linear_names
#
#     read_files, file_names = select_files.select(Relation, Threshold, Test_flies_num, seed=test_seed)
#
#
#
#     # values_return = return_values(read_files, pre_process=Pre_process, reverse=False, thread=Thread, seed=i, method='', noise_level=n, add_on=Add_on, show_fig=False)
#
#     values_return = return_values(file_value_list=read_files, reverse=False, thread=Thread, pre_process=Pre_process,
#                                   seed=test_seed,
#                                   method=Method, **Kwargs)
#
#     values_Re_return = return_values(file_value_list=read_files, reverse=True, thread=Thread, pre_process=Pre_process,
#                                      seed=test_seed,
#                                      method=Method, **Kwargs)
#
#     result_DF = pd.concat([values_return, values_Re_return], axis=1)
#     result_DF.index = file_names
#
#     if os.path.exists(rf'analysis/{Method}'):
#         pass
#     else:
#         os.mkdir(rf'analysis/{Method}')
#         print(f'已创建临时文件夹 analysis/{Method}')
#
#     result_DF.to_excel(
#         rf'analysis/{Method}/{Relation}_{Threshold[0]}-{Threshold[1]}_{Test_flies_num}-CE_{Pre_process_name}{Kwargs_name}_{Method}.xlsx')