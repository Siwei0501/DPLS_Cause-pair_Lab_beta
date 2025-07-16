import math
import time
from functools import wraps
from typing import Literal, Union, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from numpy import ndarray, dtype
from pandas import DataFrame
from sklearn.svm import SVR


def SVR_cv_tester(x_svr, y_svr, C=1e2, kernel='rbf', gamma=0.702, cv=10, **kwargs):
    R2_list_X = []

    x_svr = np.array(x_svr)
    y_svr = np.array(y_svr)

    if y_svr.ndim == 1:
        y_svr = y_svr.reshape(-1, 1)
    if x_svr.ndim == 1:
        x_svr = x_svr.reshape(-1, 1)

    train_SVR_index, test_SVR_index = spliter(x_svr, cv=cv)

    for j in range(cv):
        test_y_svr = y_svr[test_SVR_index[j], :]
        train_y_svr = y_svr[train_SVR_index[j], :]

        test_x_svr = x_svr[test_SVR_index[j], :]
        train_x_svr = x_svr[train_SVR_index[j], :]

        # 默认情况：SVR_kernel='rbf', SVR_C=1e2, SVR_gamma=0.8
        svr_rbf_X = SVR(kernel=kernel, C=C, gamma=gamma, cache_size=1024)

        svr_rbf_X.fit(train_x_svr, train_y_svr.flatten())
        y_predict = svr_rbf_X.predict(np.array(test_x_svr))

        R2_X1 = (np.corrcoef(test_y_svr.reshape(-1), y_predict.reshape(-1)) ** 2)[0, 1]

        R2_list_X.append(R2_X1)

    return R2_list_X, np.mean(R2_list_X)

# 测试集与训练集分割器
def spliter(sample_num, cv: int = 5, mode: Literal['uniform', 'layer'] = 'layer', random_before: bool = False, shuffle_seed=None, **kwargs):
    """
    :param sample_num: 样本数
    :param cv:  分割折数
    :param mode: 'layer':分层分割， ‘uniform’：均匀分割
    :param random_before: 是否在分割前随机打乱样本序号
    :return:  train_list[train1_id, train2_id, ..., train_cv_id], test_list[test1_id, test2_id, ..., test_cv_id],
    """

    if cv <= 1:
        return [list(range(sample_num))], [list(range(sample_num))]

    sample_index = np.arange(sample_num)
    train_list = []
    test_list = []

    if random_before:

        np.random.seed(shuffle_seed)
        np.random.shuffle(sample_index)

    if mode == 'layer':

        for i in range(cv):
            test_index = range(i, sample_num, cv)
            test_index = list(sample_index[[test_index]].reshape(-1))

            train_index = list(set(sample_index.tolist()) - set(test_index))

            train_list.append(train_index)
            test_list.append(test_index)

        return train_list, test_list

    if mode == 'uniform':

        each = sample_num // cv

        for i in range(cv):

            test_index = sample_index[i * each: (i + 1) * each].tolist()

            if i == (cv - 1):
                test_index = sample_index[i * each: sample_num].tolist()

            train_index = list(set(sample_index.tolist()) - set(test_index))

            train_list.append(train_index)
            test_list.append(test_index)

        return train_list, test_list

def do_classify(classify_name, file_values:pd.DataFrame, train_list: list, test_list: list, **kwargs):

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV

    classify_dict = {
        "Logistic_Regression": LogisticRegression(),
        "Decision_Tree": DecisionTreeClassifier(),
        "Random_Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),  # 设置 probability=True 以支持 predict_proba
        "KNN": KNeighborsClassifier(),
        "Naive_Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),

    }

    if classify_name in classify_dict:
        classify_model = classify_dict[classify_name]

    else:
        return 0

    param_grid_dict = {
        "Logistic_Regression": {
            'C': [0.1, 1, 10],
            'max_iter': [100, 1000],

        },
        "Decision_Tree": {
            'max_depth': [3, 5, 10],

            'min_samples_split': [2,10],
        },
        "Random_Forest": {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 10],

        },
        "SVM": {
            'C': [0.1, 1, 10],
        },
        "KNN": {
            'n_neighbors': [3, 5, 7],

        },
        "Naive_Bayes": {},  # 没有超参数
        "LDA": {
            "solver": ["svd", "lsqr", "eigen"],
            "shrinkage": ["auto", "float"],

        }  # 没有超参数
    }

    param_grid = param_grid_dict.get(classify_name, {})

    for k, v in kwargs.items():
        param_grid.setdefault(k, v)

    if 'parameter_optimization' in param_grid:
        del param_grid['parameter_optimization']



    X = stdize(file_values.iloc[:, :-1].copy())
    y = file_values.iloc[:, -1]



    X = to_2D_ary(X)
    y = np.array(y).flatten()

    accuracy_of_preds = []

    for train_idx, test_idx in zip(train_list, test_list):

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        try:

            if kwargs['parameter_optimization']:

                grid = GridSearchCV(classify_model, param_grid, cv=5, scoring='accuracy', n_jobs=1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)

            else:

                model_obj = classify_model.fit(X_train, y_train)
                y_pred = model_obj.predict(X_test)

        except KeyError:

            model_obj = classify_model.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)


        accuracy = accuracy_score(y_test, y_pred)
        accuracy_of_preds.append(accuracy)

    return accuracy_of_preds





def find_duplicate_columns(host_array, guest_array):
    """
    查找array_A中与array_B重复的列，并返回它们在array_A中的索引。

    参数:
        host_array: NumPy 2D 数组
        guest_array: NumPy 2D 数组

    返回:
        list: array_A中重复列的索引列表。如果没有重复列，则返回False。
    """
    # 使用NumPy的广播机制比较两数组的每一列
    # 扩展 host_array 和 guest_array，使它们的形状便于逐列比较
    A_expanded = host_array[:, :, np.newaxis]  # 扩展 host_array 的列
    B_expanded = guest_array[:, np.newaxis, :]  # 扩展 guest_array 的列

    # 比较 host_array 的列与 guest_array 的列
    equal_columns = np.all(A_expanded == B_expanded, axis=0)

    # 找到 host_array 中与 guest_array 中重复的列
    duplicate_indices = np.where(np.any(equal_columns, axis=1))[0]

    # 如果有重复列，返回其索引，否则返回 False
    return duplicate_indices.tolist() if duplicate_indices.size > 0 else False


def stdize(data: DataFrame | np.ndarray,
           nan_mode: Literal['no', '0', 'mean',] = 'no',
           mode: Literal['local', 'global'] = 'local',
           need_hot_spot: object = False, std_limit: object = 0.0001,
           axis: int = 1,
           global_mean: Union[float, list] = None,
           global_std: Union[float, list] = None,
           **kwargs: object,
           ) -> tuple[DataFrame, DataFrame] | DataFrame | tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]] | ndarray[Any, dtype[Any]]:
    """

    :param data: 输入数据
    :param nan_mode: 对数据中空值的处理方式， 'adjust': 自适应 (非空:(matrix-nan_mean)/nan_std, 空值:仅呈递). ’0‘：设定为0， ’mean‘：设定为当列/行均值
    :param need_hot_spot: 需要每个特征的样本热力值
    :param std_limit: 除0保护， 当特征方差小于std_limit时，整列设置为0
    :param axis: 标准化进行的轴， 默认为按列标准化：axis=1
    :param mode: 'local':规则标准化,'global':全局标准化(根据提供的 mean 和 std 进行标准化,而非样本的)
    :param global_std: 提供的全局标准差,用于 global times_mode
    :param global_mean: 提供的全局平均值, 用于 global times_mode
    :return: 标准化后的 array 或 Dataframe， 取决于输入格式

    """

    if len(data.shape) > 2:
        raise ValueError(f"标准化过程期望输入矩阵,你的输入维度是:{len(data.shape)}")

    # 确保使用双精度浮点数
    data_ary = np.array(data, dtype='float64')

    # 确保维度为2
    if np.ndim(data_ary) == 1:
        data_ary = data_ary.reshape(-1, 1)

    # 定义参数:standard_ary. 用于存储每一次循环被标准化后的向量
    standard_ary = np.zeros_like(data_ary)
    # 定义参数: hot_spot. 用于存储每一次循环所求得的样本绝对值占总体绝对值的比例
    hot_spot = np.zeros_like(data_ary)

    for i in range(data_ary.shape[axis]):

        # slice_: 每次循环处理的单个向量
        slice_ = data_ary[:, i] if axis == 1 else data_ary[i, :]
        # nan_mask: 向量中空值的标记,用于存储空值位置
        nan_mask = np.isnan(slice_)

        # 根据 nan_mode 处理空值
        if nan_mode == '0':
            slice_[nan_mask] = 0
        elif nan_mode == 'mean':
            slice_[nan_mask] = np.nanmean(slice_)

        # n: slice_中非空的数量.
        n = np.sum(~nan_mask)

        if n >= 1:

            # 计算无偏标准差
            if mode == 'local':
                mean = np.nanmean(slice_)
                variance = np.nansum((slice_ - mean) ** 2) / (n - 1) if n > 30 else np.nansum(
                    (slice_ - mean) ** 2) / n
                std = np.sqrt(variance)

            elif mode == 'global':

                if global_mean:
                    mean = global_mean[i]
                else:
                    mean = np.nanmean(slice_)
                if global_std:
                    std = global_std[i]
                else:
                    variance = np.nansum((slice_ - mean) ** 2) / (n - 1) if n > 30 else np.nansum(
                        (slice_ - mean) ** 2) / n
                    std = np.sqrt(variance)
            else:
                raise TypeError(f"times_mode type '{mode}' not understood")

            if std > std_limit:
                standard_slice = (slice_ - mean) / std
                hot_spot_slice = np.abs(standard_slice) / np.sum(np.abs(standard_slice))
            else:
                standard_slice = slice_
                hot_spot_slice = slice_
                standard_slice[~nan_mask] = 0
                hot_spot_slice[~nan_mask] = 0

        else:
            standard_slice = np.full_like(slice_, np.nan)
            hot_spot_slice = np.full_like(slice_, np.nan)

        if axis == 1:
            standard_ary[:, i] = standard_slice.flatten()
            hot_spot[:, i] = hot_spot_slice.flatten()
        else:
            standard_ary[i, :] = standard_slice.flatten()
            hot_spot[i, :] = hot_spot_slice.flatten()

    if isinstance(data, pd.DataFrame):
        if axis == 1:
            standard_df = pd.DataFrame(standard_ary, columns=data.columns, index=data.index)
            hot_spot_df = pd.DataFrame(hot_spot, columns=data.columns, index=data.index)
        else:
            standard_df = pd.DataFrame(standard_ary, columns=data.index, index=data.columns).T
            hot_spot_df = pd.DataFrame(hot_spot, columns=data.index, index=data.columns).T
        return (standard_df, hot_spot_df) if need_hot_spot else standard_df

    return (standard_ary, hot_spot) if need_hot_spot else standard_ary


def normalize(data: DataFrame | np.ndarray,
              nan_mode: Literal['no', '0', 'mean',] = 'mean',
              mode: Literal['local', 'global'] = 'local',
              std_limit=0.0001,
              axis: int = 1,
              threshold=None,
              global_max: Union[float, list] = None,
              global_min: Union[float, list] = None,
              **kwargs
              ):

    if threshold is None:
        threshold = [-1, 1]  # 还没实现的功能, 期望能映射到任意区间

    if len(list(data.shape)) > 2:
        raise ValueError(f"标准化过程期望输入矩阵,你的输入维度是:{len(data.shape)}")

    data_ary = np.array(data, dtype='float64')  # 不管输入是什么格式, 先把他变成ary再处理

    # 确保维度为2
    if np.ndim(data_ary) == 1:
        data_ary = data_ary.reshape(-1, 1)

    # 定义参数:norm_ary. 用于存储每一次循环被标准化后的向量
    norm_ary = np.zeros_like(data_ary)

    for i in range(data_ary.shape[axis]):

        # slice_: 每次循环处理的单个向量
        slice_ = data_ary[:, i] if axis == 1 else data_ary[i, :]
        # nan_mask: 向量中空值的标记,用于存储空值位置
        nan_mask = np.isnan(slice_)

        # 根据 nan_mode 处理空值
        if nan_mode == '0':
            slice_[nan_mask] = 0
        elif nan_mode == 'mean':
            slice_[nan_mask] = np.nanmean(slice_)

        if any(~nan_mask):
            # 计算无偏标准差
            if mode == 'local':  # 如果归一化模式为局部,则把样本最大、最小值赋给 col_max, col_min
                col_max = np.nanmax(slice_)
                col_min = np.nanmin(slice_)
            elif mode == 'global':  # 如果归一化模式为全局, 则把输入的 global_max、global_min 赋给 col_max, col_min
                col_max = global_max[i]
                col_min = global_min[i]

            else:
                raise TypeError(f"times_mode type '{mode}' not understood")

            if np.nanstd(slice_) > std_limit:

                temp_col_norm = (slice_ - col_min) / (col_max - col_min)  # 👈对temp_col_array进行归一化, 归一至[0, 1]
                temp_col_norm = (temp_col_norm - 0.5) * 2  # 归一至[-1, 1]
            else:
                temp_col_norm = slice_
                temp_col_norm[~nan_mask] = 0

        else:
            temp_col_norm = np.full_like(slice_, np.nan)

        if axis == 1:
            norm_ary[:, i] = temp_col_norm.flatten()
        elif axis == 0:
            norm_ary[i, :] = temp_col_norm.flatten()
        else:
            raise ValueError(f"axis value '{axis}' not match")

    if isinstance(data, pd.DataFrame):
        norm_df = pd.DataFrame(norm_ary, columns=data.columns, index=data.index)
        return norm_df

    return norm_ary


def data_scaler(data,
                process: Literal['no', 'norm', 'stand'],
                nan_mode: Literal['no', '0', 'mean',] = 'no',
                mode: Literal['local', 'global'] = 'local',
                need_hot_spot = False,
                std_limit: float = 0.01,
                global_max: Union[float, list] = None,
                global_min: Union[float, list] = None,
                global_mean: Union[float, list] = None,
                global_std: Union[float, list] = None,
                axis=1,

                ):
    if process == 'norm':
        data_processed = normalize(data, axis=axis, nan_mode=nan_mode, mode=mode, global_max=global_max,
                                   global_min=global_min, std_limit=std_limit)
        return data_processed
    elif process == 'stand':

        if need_hot_spot:
            data_processed, data_hot_spot = stdize(data, axis=axis, nan_mode=nan_mode, mode=mode, global_std=global_std,
                                global_mean=global_min, std_limit=std_limit, need_hot_spot=need_hot_spot)
            return data_processed, data_hot_spot

        data_processed = stdize(data, axis=axis, nan_mode=nan_mode, mode=mode, global_std=global_std,
                                global_mean=global_min, std_limit=std_limit)
        return data_processed
    else:
        Warning(f"process type '{process}' not understood")
        return data


# def cv_stability(result_list: list):
#     return stability(result_list)


def distance_matrix(vectors_list:list[np.ndarray],
                    thread=1,
                    dtype='float32',
                    distance_mode:list[str] | str ='Euc', distance_r:int=2, **kwargs):


    def vector_core(vectors: np.ndarray):

        vectors = vectors.astype(dtype)

        # Ensure the input is at most 2D and one of the dimensions is 1
        if len(vectors.shape) > 2:
            raise ValueError('Distance_matrix function expects Matrix input')

        if len(vectors.shape) == 1 or len(vectors.shape) == 2 and 1 in vectors.shape :

            vectors = vectors.reshape(-1, 1)
            distance_ary = np.abs(vectors - vectors.T)

            return distance_ary

        else:

            matrix_tiled = np.tile(vectors, (vectors.shape[0], 1, 1))
            # distance_ary_tiled = np.linalg.norm(matrix_tiled - vectors.reshape(vectors.shape[0], 1, vectors.shape[1]), axis=2)

            distance_ary_tiled = np.abs(matrix_tiled - vectors.reshape(vectors.shape[0], 1, vectors.shape[1])) ** distance_r
            distance_ary_tiled = np.sum(distance_ary_tiled, axis=2) ** (1 / distance_r)

            distance_ary = distance_ary_tiled.reshape(vectors.shape[0], vectors.shape[0])

            return distance_ary
 
    def tensor_core(matrix: np.ndarray):

        matrix = matrix.astype(dtype)

        # Ensure the input is at most 2D and one of the dimensions is 1
        if len(matrix.shape) > 2:
            raise ValueError('Distance_matrix function expects Matrix input')

        if len(matrix .shape) == 1 or len(matrix .shape) == 2 and 1 in matrix .shape :

            distance_ary = vector_core(matrix)
            return distance_ary.astype(dtype)

        else:

            matrix_tiled = np.tile(matrix, (1, matrix.shape[0]))
            matrix_flatten = matrix.flatten()
            matrix_flatten = np.tile(matrix_flatten, (matrix_tiled.shape[0], 1))

            return_matrix = np.abs(matrix_tiled - matrix_flatten)

            return return_matrix

    if distance_mode == 'Euc':
        func = vector_core
    elif distance_mode == 'Mah':
        func = tensor_core
    else:
        raise ValueError(f"distance_mode '{distance_mode}' not understood")

    if thread == 1:
        result_list = []
        for vectors_ in vectors_list:
            result_list.append(func(vectors_))
        return result_list

    else:

        result_tuple = Parallel(n_jobs=thread)(delayed(func)(vectors_, dtype) for vectors_ in vectors_list)
        return result_tuple



def time_used(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算用时
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds")
        return result  # 返回函数的结果

    return wrapper


def color_seper(theme:int = 2, sub=(3,3), show=False,
                gen_mode:Literal['centre', 'top', 'bottom']='centre', seed=0):

    import colorsys

    def generate_contrasting_colors(theme_: int):
        """
        生成对比鲜明的颜色代码。

        参数:
            theme_ (int): 要生成的颜色数量。

        返回:
            list: 包含十六进制颜色代码的列表。
        """


        # 确保 theme_ 至少为 1
        theme_ = max(1, theme_)
        np.random.seed(seed)
        # 将色相值均匀分布
        colors = []
        for t in range(theme_):
            hue = np.sin(2*np.pi*(t / theme_))  # 色相均匀分布在 [0, 1] 区间
            lightness = abs(0.7 + np.random.normal(0,0.2,1)[0]) # 保持中等亮度
            saturation = abs(0.7 + np.random.normal(0,0.2,1)[0]) # 较高的饱和度

            if lightness > 1:
                lightness = 0.7
            if saturation > 1:
                saturation = 0.7

            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            # 转换为十六进制格式
            hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            colors.append(hex_color)

        return colors

    def generate_color_gradient(color: str, num: int, mode: str = 'centre'):
        """
        根据基准颜色生成颜色梯度。

        参数:
            color (str): 基准颜色的十六进制代码（例如 '#ff5733'）。
            num (int): 要生成的颜色数量。
            times_mode (str): 'centre' | 'top' | 'bottom'，控制颜色渐变的方向。

        返回:
            list: 包含生成的颜色梯度的十六进制颜色代码列表。
        """
        import colorsys

        # 将十六进制颜色转换为RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        # 将RGB转换为十六进制颜色
        def rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(*rgb)

        # 获取基准颜色的RGB和HSL
        r, g, b = hex_to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)

        gradients = []

        # 生成梯度
        for n in range(num):
            if mode == 'centre':
                # 在中心对称分布亮度，范围为 [l-0.3, l+0.3]
                factor = (n - (num - 1) / 2) / ((num - 1) / 2) if num > 1 else 0
                new_l = min(max(l + factor * 0.3, 0), 1)
            elif mode == 'top':
                # 从基准颜色逐渐变淡（增加亮度）
                factor = n / (num - 1) if num > 1 else 0
                new_l = min(max(l + factor * (1 - l), 0), 1)
            elif mode == 'bottom':
                # 从基准颜色逐渐变深（降低亮度）
                factor = n / (num - 1) if num > 1 else 0
                new_l = min(max(l - factor * l, 0), 1)
            else:
                raise ValueError("times_mode must be 'centre', 'top', or 'bottom'")

            # 将调整后的HSL转换回RGB并存储
            new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
            gradients.append(rgb_to_hex((int(new_r * 255), int(new_g * 255), int(new_b * 255))))

        return gradients

    def display_colors(colors: dict):
        """
        显示多组颜色，每组以基准颜色为标题。

        参数:
            colors (dict): 字典格式，键为基准颜色，值为与之相关的颜色列表。
        """
        num_groups = len(colors)
        max_colors_in_group = max(len(v) for v in colors.values())

        fig, axs = plt.subplots(num_groups, max_colors_in_group + 1,
                                figsize=(2 * (max_colors_in_group + 1), 2 * num_groups))

        # 如果只有一组颜色，保证 axs 是可迭代的二维数组
        if num_groups == 1:
            axs = [axs]

        # 遍历每组基准颜色及其对应的颜色列表
        for row,  gradient_colors in enumerate(colors.values()):
            # 设置基准颜色
            # axs[row][0].set_facecolor(base_color)
            # axs[row][0].set_title(base_color, fontsize=10, pad=10)
            axs[row][0].set_xticks([])
            axs[row][0].set_yticks([])
            axs[row][0].spines['top'].set_visible(False)
            axs[row][0].spines['bottom'].set_visible(False)
            axs[row][0].spines['left'].set_visible(False)
            axs[row][0].spines['right'].set_visible(False)

            # 设置对应的颜色梯度
            for col, color in enumerate(gradient_colors):
                axs[row][col + 1].set_facecolor(color)
                axs[row][col + 1].set_xticks([])
                axs[row][col + 1].set_yticks([])
                axs[row][col + 1].spines['top'].set_visible(False)
                axs[row][col + 1].spines['bottom'].set_visible(False)
                axs[row][col + 1].spines['left'].set_visible(False)
                axs[row][col + 1].spines['right'].set_visible(False)

            # 隐藏多余的空白子图
            for col in range(len(gradient_colors) + 1, max_colors_in_group + 1):
                axs[row][col].axis('off')

        plt.tight_layout()
        plt.show()


    color_dict = {}
    theme_color = generate_contrasting_colors(theme_=theme)

    for i, theme_i in enumerate(theme_color):

        color_dict[i] = generate_color_gradient(color=theme_i, num=sub[i], mode=gen_mode)

    if show:

        display_colors(color_dict)

    return color_dict


def calculate_corr(X, y, cv=1, R2=True, independent=True, detail=False, **kwargs):

    X = to_2D_ary(X)
    y = to_2D_ary(y)

    train_set, test_set = spliter(X.shape[0], cv=cv)

    X_centre = X - np.nanmean(X, axis=0)
    y_centre = y - np.nanmean(y, axis=0)

    X_std = np.nanstd(X, axis=0)
    y_std = np.nanstd(y, axis=0)

    # 每个特征的 s_X * s_y
    X_s = (X_std * y_std).reshape(-1, 1)

    R_list = []

    for train_i in train_set:

        if independent:

            X_cv_i = X[train_i, :]
            y_cv_i = y[train_i, :]

            X_cv_i = X_cv_i - np.nanmean(X_cv_i, axis=0)
            y_cv_i = y_cv_i - np.nanmean(y_cv_i, axis=0)

            X_std_i = np.nanstd(X_cv_i, axis=0)
            y_std_i = np.nanstd(y_cv_i, axis=0)

            X_s_i = (X_std_i * y_std_i).reshape(-1, 1)

        else:

            X_cv_i = X_centre[train_i, :]
            y_cv_i = y_centre[train_i, :]

            X_s_i = X_s

        Cov_i = (X_cv_i.T @ y_cv_i) / X_cv_i.shape[0]

        if R2:
            R = (Cov_i / X_s_i) ** 2
        else:
            R = Cov_i / X_s_i

        R = np.nan_to_num(R, nan=0, posinf=0, neginf=0)
        R_list.append(R.flatten())

    if detail:
        return np.array(R_list).T

    return np.mean(R_list, axis=0).tolist()


def to_2D_ary(matrix: Union[DataFrame, np.ndarray]) -> np.ndarray:

    """
    接受: 一维向量或矩阵
    :param vectors:
    :return: 2维向量或矩阵

    """
    if len(matrix.shape) > 2:
        raise f'to_2D_ary except 2 dimension data'

    if len(matrix.shape) == 1:
        return np.array(matrix).reshape(-1, 1)
    else:
        return np.array(matrix)


def calculate_alpha(sample_num, alpha, tail='two'):
    """
    计算在给定样本量和显著性水平下所需的相关系数 R。

    :param sample_num: 样本量
    :param alpha: 显著性水平（如0.05）
    :param tail: 检验类型 ('two' 表示双尾，'one' 表示单尾)
    :return: 所需的相关系数 R
    """
    # 计算自由度
    df = sample_num - 2

    # 根据检验类型选择临界 t 值
    if tail == 'two':
        # 双尾检验：alpha/2
        t_critical = stats.t.ppf(1 - alpha / 2, df)
    elif tail == 'one':
        # 单尾检验：alpha
        t_critical = stats.t.ppf(1 - alpha, df)
    else:
        raise ValueError("Invalid 'tail' argument. Choose either 'two' or 'one'.")

    # 计算所需的 R
    p_value = t_critical / math.sqrt(t_critical ** 2 + df)

    return p_value


if __name__ == '__main__':

    combined_data = pd.read_excel('4000-02.xlsx', header=0, index_col=0)
    combined_data = combined_data.dropna(axis=0, how='any')
    train_list, test_list = spliter(combined_data.shape[0], cv=5)


    classify_ = [
        "Logistic_Regression",
        "Decision_Tree",
        "Random_Forest",
        "SVM",  # 设置 probability=True 以支持 predict_proba
        "KNN",
        "Naive_Bayes","LDA"
    ]
    from tqdm import tqdm
    classify_results = {}
    for classif in tqdm(classify_):

        classify_result_i = do_classify(classify_name=classif, file_values=combined_data,test_list=train_list,train_list=test_list, parameter_optimization=True)
        classify_results[classif] = classify_result_i

    results_DF = pd.DataFrame.from_dict(classify_results, orient='index')
    results_DF.to_excel('CE_4000-02_result.xlsx')
