import importlib
import os

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from joblib import Parallel, delayed
from stqdm import stqdm
from typing import Literal, Union, Callable

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import io
import zipfile
import copy


function_dict = {
    "线性函数": None,
    "正弦函数": lambda f: np.cos(np.pi * f),
    "余弦函数": lambda f: np.cos(np.pi * f),
    "二次函数": lambda f: 2 * (f ** 2),
    "平方根函数": lambda f: np.sqrt(np.abs(f)),
    "指数函数": lambda f: np.exp(f),
    "对数函数（平移）": lambda f: np.log(f + 1),
    "对数函数（加偏移防负值）": lambda f: np.log(np.abs(f) + 1e-10),
    "Sigmoid 函数": lambda f: 1 / (1 + np.exp(-6 * f)),
    "三次多项式函数": lambda f: 2 * (f ** 3) + f ** 2 - 2 * f,
    "指数幂函数": lambda f: 2 ** (f + 1),
    "高频正弦函数": lambda f: np.sin(6 * np.pi * f),
    "混合三角+线性函数": lambda f: 0.2 * np.sin(4 * f) + (11 / 10) * f,
    "高频正弦 + 线性项": lambda f: np.sin(5 * np.pi * f) + f,
    "高频余弦函数": lambda f: np.cos(6 * np.pi * f),
    "高频正弦线性混合函数": lambda f: (1 / 10) * np.sin(10.6 * f) + (11 / 10) * f,
    "非线性频率余弦函数": lambda f: np.cos(5 * np.pi * f * (f + 1)),
    "非线性频率正弦函数": lambda f: np.sin(4 * np.pi * f * (f + 1)),
}


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


def cal_DPLS_obj(file_value:pd.DataFrame, **kwargs):

    value_copy = file_value.copy()

    DPLS_obj = DPLS(**kwargs).fit(value_copy[[kwargs['reason']]], value_copy[[kwargs['result']]], **kwargs)

    return DPLS_obj


# 4-2 方法参数映射（用于动态显示对应的子参数控件）
DPLSR_param_dict = {
    "cv": {
        "type": "slider",
        "label": "多重检验折数",
        "min": 1,
        "max": 10,
        "step": 1,
        "value": 5
    },

    "max_iter": {
        "type": "slider",
        "label": "DPLS最大迭代层",
        "min": 1,
        "max": 500,
        "value": 20
    },

    "R_mode": {
        "type": "select_slider",
        "label": "求R模式",
        "help": "[fusion]: 返回整个样本集的DPLSR, [single]: 返回每列的DPLSR",
        "options": ['fusion', 'single'],
        "value": 'fusion'
    },

    "fit_mode": {
        "type": "selectbox",
        "label": "Fit-矫正模式",
        "help": "[Fit]: 纯 Fit, [CV]: 纯 CV, [Fit_rectify]: CV 结果用 Fit 矫正",
        "options": ["Fit", 'CV', 'Fit_rectify'],
        "value": 'CV'
    },

    "distance_pattern": {
        "type": "multiselect",
        "label": "核函数",
        "help": "[Euc]:欧氏距离. [Mah]:曼哈顿距离, [Pairs]:成对组合距离, [Ming]:闵氏距离",
        "options": ["Euc", 'Mah', 'Pairs', 'Ming'],
        "value": "Euc"
    },


    "fit_intercept": {
        "type": "checkbox",
        "label": "截距项",
        "value": False
    },

    "whiten": {
        "type": "checkbox",
        "label": "标准化",
        "value": False
    },
    "square": {
        "type": "checkbox",
        "label": "距离矩阵左乘自己的转置",
        "value": True
    }
}

# 3-1 描述预处理
preprocess_descriptions = {

    "normalize": "标准化",
    "regionalitze": "区域化,默认区域[-1,1]",
    "Sort_by_reason": "样本根据原因升序排列",
    "Xmean": "把X分组, 每组取平均值,以此获得平滑的新X",
    "drop_Bias":"去除偏倚值",
    "drop_NA":"去除空值",
    "drop_duplicates_mean": "去除 reason 重复数据, result 取均值",
    "add_noise": "对数据添加随机噪声",
    "subsidiary_sampling": "组子采样, 把样本升序排列后分割成等宽的小组, 在每个小组内抽一个组成新样本集",

    "to_DPLS_pred": "把 result 替换为 DPLS 预测值",
    "None": "无预处理",

}

# 3-2 预处理的参数映射
preprocess_param_controls = {

    "normalize": {
        'nan_mode': {"type": "selectbox", "label": "空值应对策略",
                     "options": ['no', '0', 'mean'], "value": 'no', "help": "[no]: 仅呈递, [0]: 替换为0, [mean]: 替换为均值"},

        'axis': {"type": "selectbox", "label": "方向", "help":"[1]: 按列标准化, [0], 按行标准化",
                 "options": [1, 0], "value": 1},
    },

    "regionalitze": {
        'nan_mode': {"type": "selectbox", "label": "空值应对策略","help":"[no]: 仅呈递, [0]: 替换为0, [mean]: 替换为均值",

                     "options": ['no', '0', 'mean'], "value": 'no'},

        'axis': {"type": "selectbox", "label": "方向","help":"[1]: 按列归一化, [0], 按行归一化",
                 "options": [1, 0], "value": 1},
    },

    "add_noise": {
        'noise_level': {"type": "slider", "label": "噪音强度", "min": 0.0, "max": 10.0, "value": 0.5, "step": 0.05},
        'add_on': {"type": "selectbox",
                   "label": "噪音添加位置","help":"[reason]: 噪音加在 reason 上, [result], 噪音加在 result 上",
                   "options": ['reason', 'result'], "value": 'result'},
        'noise_mode': {"type": "selectbox", "label": "噪音模式","help":"[normal]: 添加高斯噪音, [uniform], 添加均匀噪音",
                       "options": ['normal', 'uniform'], "value": 'normal'},
        'adjust': {"type": "checkbox", "label": "自适应噪音强度", "value": False, "help":"( noise_level *= DPLSR^2 )"},

    },

    "subsidiary_sampling": {
        'subsidiary_width': {"type": "slider", "label": "组大小", "min": 2, "max": 5, "value": 2, "step": 1},
    },

    "to_DPLS_pred": {} | DPLSR_param_dict,

    "Xmean": {
    "mean_ratio": {
        "type": "slider",
        "label": "分组率, 分组越多则平滑程度越低",
        "min": 0.1, "max": 1.0, "step": 0.1, "value": 0.5
    },
    "Xmean_mode": {
        "type": "selectbox",
        "label": "分组模式",
        "help": "[uniform]: 每组宽度相等, [normal]: 每组概率相等",
        "options": ['normal', 'uniform'],
        "value": 'normal'
    },
    "Xmean_window": {
        "type": "slider",
        "label": "窗口长度, 即'勾搭臂长'",
        "min": 1, "max": 10, "step": 1, "value": 2
    },
},

    "drop_Bias": {
    "Bias_threshold": {
        "type": "slider",
        "label": "界定偏倚值的σ阈值",
        "min": 1.5, "max": 5.0, "step": 0.1, "value": 3.0
    },
}

}

# 4-1 方法的描述
method_descriptions = {

    "DPLS": "返回需要的 DPLS 实例属性",
    "DPLSR": "计算 DPLSR ",
    "PATH": "计算路径增量 Path delta",

    "P_DPLSR": "计算指定 P 的 DPLS",
    "DPLS_predR": "计算 reason 与 result_pred 之间的 DPLSR。",
    "chain_stability": "多轮链式 DPLS 拟合，标准差衡量因果路径波动。",
    "break_DPLSR": "将数据升序排列再分段后分别拟合，统计各段DPLSR之方差用于评估稳定性。",
    'PersonR': "皮尔逊系数",
    'variable_coefficient': "变异系数, CV = 方差/均值",

    "PATH_tender": "不同滑动窗口下的 PATH 值序列，观察路径稳定性。",
    "shuffle_path": "通过混淆样本原顺序, 改变重复值的排列, 提高估计路径结构稳定性。-脆弱的分析, 运行需要稳定的数据",
    "GS": "使用地统计学核分类的 PATH",

    "DPLSe_KCI": "DPLS(reason, result)的残差与的 reason 的 KCI 值",
    "DPLSe_P_KCI": "计算指定 P 的 DPLS(reason, result) 的残差与的 reason 的 KCI 值",
    'DPLSe_HSIC': "DPLS(reason, result)的残差与的 reason 的 HSIC 值",
    "CMVe_DPLSR": "先排除 reason, result 内协变量的非线性影响, 再计算 reason_clear, result_clear 之间的 DPLSR.",
    'CMVe_DPLSe_KCI': "先排除 reason, result 内协变量的非线性影响, 再计算 reason_clear, result_clear 之间的 DPLS 残差与 reason_clear 的 KCI。",
    'aCMV_DPLSe_KCI': "先求出协变量, 把协变量和 reason 按列拼接: a_cmv=[reason,cmv], 再计算 a_cmv 和 result 之间的 DPLS 残差与 a_cmv 的 KCI。",

    "is_Linear": "评估因果关系是否近似线性，返回双向相关系数。",

}


PATH_param_dict = {
    "Path_centre": {"type": "checkbox", "label": "中心化后再求 PATH", "value": False},
    "Path_normal": {"type": "checkbox", "label": "PATH 除以输入样本的标准差", "value": False},
    "Path_window": {"type": "slider", "label": "delta间隔", "min": 1, "max": 10, "value": 1},
    "Sort_by": {"type": "selectbox",
                "label": "排序模式", "help":"[reason]: 仅根据 reason 排序,重复值保留原始顺序, [all]: result 参与排序,作为排序的次级依据",
                "options": ["all", "reason", ], "value": "all"},
}


DPLS_needed_param = ["max_iter", "R2", "cv_R2", "fit_R2", "p", "cv_p", "fit_p", "y_pred_R2",]
DPLS_attr_dict = {f"_{k}_": {"type": "checkbox", "label":f"_{k}_", "value": False, "inner_col": e%2} for e, k in enumerate(DPLS_needed_param)}

print("DPLS_attr_dict", DPLS_attr_dict)


method_param_controls = {

    "DPLS": DPLS_attr_dict | DPLSR_param_dict,

    "DPLSR": {} | DPLSR_param_dict,

    "P_DPLSR": {
        "P_mode": {
            "type": "selectbox",
            "label": "选择P类型",
            "help": "在P_AB,P_BA中, [min]: P_min, [max]:P_max, [mean]: P_mean",
            "options": ['min', 'max', 'mean'],
            "value": "mean"
        }
    } | DPLSR_param_dict,

    "DPLS_predR": {} | DPLSR_param_dict,

    "PATH": {} | PATH_param_dict,

    "GS": {
        "GS_core": {
            "type": "selectbox",
            "label": "核函数类型",
            "options": ["Matern"],
            "value": "Matern"
        },
        "GS_ratio": {
            "type": "slider",
            "label": "比例参数, 0.5倍极差内的样本数 n^value",
            "min": 0.1,
            "max": 1.0,
            "step": 0.1,
            "value": 0.6
        }
    } | PATH_param_dict,

    "shuffle_path": {
        "Shuffle_times": {
            "type": "slider",
            "label": "混淆次数",
            "min": 10,
            "max": 2000,
            "step": 10,
            "value": 1000
        }
    } | PATH_param_dict,

    "break_DPLSR": {
        "break_parts": {
            "type": "slider",
            "label": "断点分段数",
            "min": 2,
            "max": 10,
            "step": 1,
            "value": 3
        }
    } | DPLSR_param_dict,

    "chain_stability": {
        "Chain_mode": {
            "type": "selectbox",
            "label": "链式模式",
            "options": ["flow", "tree"],
            "value": "flow"
        },
        "Chain_len": {
            "type": "slider",
            "label": "链长度",
            "min": 2,
            "max": 10,
            "step": 1,
            "value": 3
        }
    } | DPLSR_param_dict,

    "PATH_tender": {
        "Tender_length": {
            "type": "slider",
            "label": "倾向分析",
            "min": 2,
            "max": 10,
            "step": 1,
            "value": 5
        }
    } | PATH_param_dict,

    "PersonR": {
        "R2": {
            "type": "checkbox",
            "label": "返回R^2",
            "value": True
        }
    },

    "DPLSe_KCI": {} | DPLSR_param_dict,

    "DPLSe_P_KCI": {
        "P_mode": {
            "type": "selectbox",
            "label": "选择P类型",
            "help": "在P_AB,P_BA中, [min]: P_min, [max]:P_max, [mean]: P_mean",
            "options": ['min', 'max', 'mean'],
            "value": "mean"
        }
    } | DPLSR_param_dict,

    "CMVe_DPLSR": {
        "CMV_mode": {
            "type": "selectbox",
            "label": "协变量来源",
            "help": "[distance]: 来自A&B构成的距离矩阵, [origin]: 来自A&B原始矩阵",
            "options": ['distance', 'origin'],
            "value": 'distance'
        },
        "CMV_num": {
            "type": "slider",
            "label": "协变量数量 (距离矩阵的前n个主成分)",
            "min": 1,
            "max": 50,
            "value": 1
        }
    } | DPLSR_param_dict,

    "CMVe_DPLSe_KCI": {
        "CMV_mode": {
            "type": "selectbox",
            "label": "协变量来源",
            "help": "[distance]: 来自A&B构成的距离矩阵, [origin]: 来自A&B原始矩阵",
            "options": ['distance', 'origin'],
            "value": 'distance'
        },
        "CMV_num": {
            "type": "slider",
            "label": "协变量数量 (距离矩阵的前n个主成分)",
            "min": 1,
            "max": 50,
            "value": 1
        }
    } | DPLSR_param_dict,

    "aCMV_DPLSe_KCI": {
        "CMV_mode": {
            "type": "selectbox",
            "label": "协变量来源",
            "help": "[distance]: 来自 A&B 的融合距离矩阵, [origin]: 来自 A&B 原始矩阵拼接",
            "options": ['distance', 'origin'],
            "value": 'distance'
        },
        "CMV_num": {
            "type": "slider",
            "label": "协变量数量 (距离矩阵的前n个主成分)",
            "min": 1,
            "max": 50,
            "value": 1
        }
    } | DPLSR_param_dict,

    "is_Linear": {
        "Test_times": {
            "type": "slider",
            "label": "测试次数",
            "help": "is_Linear的判断基于多次测试",
            "min": 1,
            "max": 20,
            "value": 5
        },
        "Test_ratio": {
            "type": "slider",
            "label": "测试比例",
            "help": "每次测试抽取的样本数占总样本的比例",
            "min": 0.1,
            "max": 0.9,
            "step": 0.05,
            "value": 0.7
        }
    } | DPLSR_param_dict,

}

# 5-1 分类器的描述
classify_description = {
    "Logistic_classifyn": "用于二分类或多分类问题，通过逻辑函数输出概率值，简单高效，适合线性可分数据。",

    "Decision_Tree": "通过构建树结构进行决策，易于理解和可视化，能够处理非线性关系但容易过拟合。",

    "Random_Forest": "集成多个决策树，通过投票方式提高准确率，减少过拟合，适合高维数据。",

    "SVM": "支持向量机用于寻找最佳分类边界，支持非线性分类（核函数），在高维空间表现良好。",

    "KNN": "K近邻通过测量样本之间的距离，找出最近的K个邻居进行投票分类，适合小数据量问题。",

    "Naive_Bayes": "基于贝叶斯定理和特征条件独立假设，计算效率高，适合文本分类等高维稀疏数据。",

    "LDA": "线性判别分析，通过最大化类间方差与最小化类内方差来进行分类，也常用于降维。",

}

# 5-2 引用分类器
classify_dict = {
    "Logistic_Regression": LogisticRegression(),
    "Decision_Tree": DecisionTreeClassifier(),
    "Random_Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),  # 设置 probability=True 以支持 predict_proba
    "KNN": KNeighborsClassifier(),
    "Naive_Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),

}

# 5-3 分类器参数映射
classify_param_control = {

    "Logistic_classifyn": {
        "C": {"type": "slider", "label": "正则化强度 C", "min": 0.01, "max": 10.0, "value": 1.0, "step": 0.01},
        "max_iter": {"type": "slider", "label": "最大迭代次数", "min": 100, "max": 10000, "value": 1000, "step": 100},
        "penalty": {"type": "selectbox", "label": "正则化类型", "options": ["l2", "l1", "elasticnet", "none"],
                    "value": "l2"},
        "solver": {"type": "selectbox", "label": "优化器", "options": ["lbfgs", "liblinear", "saga", "newton-cg"],
                   "value": "lbfgs"},

        "parameter_optimization":{"type": "checkbox", "label": "自动参数寻优", "value":True },

    },

    "Decision_Tree": {
        "max_depth": {"type": "slider", "label": "最大深度", "min": 1, "max": 30, "value": None},
        "min_samples_split": {"type": "slider", "label": "最小划分样本数", "min": 2, "max": 20, "value": 2},
        "criterion": {"type": "selectbox", "label": "划分标准", "options": ["gini", "entropy", "log_loss"],
                      "value": "gini"},
        "parameter_optimization":{"type": "checkbox", "label": "自动参数寻优", "value":True },
    },

    "Random_Forest": {
        "n_estimators": {"type": "slider", "label": "树的数量", "min": 10, "max": 300, "value": 100, "step": 10},
        "max_depth": {"type": "slider", "label": "最大深度", "min": 1, "max": 30, "value": None},
        "min_samples_split": {"type": "slider", "label": "最小划分样本数", "min": 2, "max": 20, "value": 2},
        "criterion": {"type": "selectbox", "label": "划分标准", "options": ["gini", "entropy", "log_loss"],
                      "value": "gini"},
        "parameter_optimization":{"type": "checkbox", "label": "自动参数寻优", "value":True },
    },

    "SVM": {
        "C": {"type": "slider", "label": "惩罚参数 C", "min": 0.1, "max": 100.0, "value": 1.0, "step": 0.1},
        "kernel": {"type": "selectbox", "label": "核函数类型", "options": ["rbf", "linear", "poly", "sigmoid"],
                   "value": "rbf"},
        "gamma": {"type": "selectbox", "label": "核系数 gamma", "options": ["scale", "auto"], "value": "scale"},
        "parameter_optimization":{"type": "checkbox", "label": "自动参数寻优", "value":True },
    },

    "KNN": {
        "n_neighbors": {"type": "slider", "label": "邻居数量", "min": 1, "max": 20, "value": 5},
        "weights": {"type": "selectbox", "label": "加权策略", "options": ["uniform", "distance"], "value": "uniform"},
        "metric": {"type": "selectbox", "label": "距离度量", "options": ["minkowski", "euclidean", "manhattan"],
                   "value": "minkowski"},
        "parameter_optimization":{"type": "checkbox", "label": "自动参数寻优", "value":True },
    },

    "Naive_Bayes": {
        # GaussianNB参数少，通常无需调参，可作为 baseline
    },

    "LDA": {
        "solver": {"type": "selectbox", "label": "求解器", "options": ["svd", "lsqr", "eigen"], "value": "svd"},
        "shrinkage": {"type": "selectbox", "label": "收缩策略", "options": [None, "auto", "float"], "value": None},
        "parameter_optimization": {"type": "checkbox", "label": "自动参数寻优", "value": True},
    },

}

# 多重检验分折器
def spliter(sample_num, cv: int = 5, mode: Literal['uniform', 'layers'] = 'layers', random_before: bool = False,
            shuffle_seed=None, **kwargs):
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

    if mode == 'layers':

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


# 并行 PersonR 求解器
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


# 数据格式规范器, 转为 2D-array
def to_2D_ary(matrix: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
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


# 单种子发散器
def gen_seed(param_num: int, rand_seed: int, gen_times: int = 1):
    np.random.seed(rand_seed)
    seed_list = np.random.randint(0, 50000, [gen_times, param_num]).reshape(gen_times, param_num)
    np.random.seed(None)
    return seed_list.tolist()


def parallel_wrapper(
        func: Callable,
        file_value_dict: dict,
        desc: str,
        thread: int = 1,
        **kwargs

) -> dict:

    return_dict = {}

    if thread == 1:

        for file_name, file_value in stqdm(file_value_dict.items(), desc=desc):
            file_return = func(file_value, **kwargs)
            return_dict[file_name] = file_return

    else:

        file_name_list = list(file_value_dict.keys())
        thread_result = Parallel(n_jobs=thread)(delayed(func)(file_value_dict[key], **kwargs) for key in file_name_list)

        for key, value in zip(file_name_list, thread_result):
            return_dict.update({key: value})

    return return_dict


def param_control_core(param_key, control, key_name):

    param_kwargs = {}

    if control["type"] == "checkbox":

        if "help" in control:

            param_kwargs[param_key] = st.checkbox(control["label"],
                                                         value=control.get("value", False),
                                                         key=key_name,
                                                         help=control["help"]
                                                         )

        else:

            param_kwargs[param_key] = st.checkbox(control["label"],
                                                         value=control.get("value", False),
                                                         key=key_name,
                                                         )


    elif control["type"] == "slider":

        if "help" in control:

            param_kwargs[param_key] = st.slider(
                control["label"],
                min_value=control["min"],
                max_value=control["max"],
                value=control["value"],
                step=control.get("step", 1),
                key=key_name,
                help=control["help"],
            )

        else:

            param_kwargs[param_key] = st.slider(
                control["label"],
                min_value=control["min"],
                max_value=control["max"],
                value=control["value"],
                step=control.get("step", 1),
                key=key_name
            )

    elif control["type"] == "selectbox":

        if "help" in control:

            param_kwargs[param_key] = st.selectbox(
                control["label"],
                control["options"],
                index=control["options"].index(control["value"]),
                key=key_name,
                help=control["help"],
            )
        else:

            param_kwargs[param_key] = st.selectbox(
                control["label"],
                control["options"],
                index=control["options"].index(control["value"]),
                key=key_name
            )

    elif control["type"] == "multiselect":

        if "help" in control:

            param_kwargs[param_key] = st.multiselect(
                control["label"],
                options=control["options"],
                default=control.get("value", []),
                key=key_name,
                help=control["help"],
            )
        else:
            param_kwargs[param_key] = st.multiselect(
                control["label"],
                options=control["options"],
                default=control.get("value", []),
                key=key_name
            )

    elif control["type"] == "select_slider":

        if "help" in control:

            param_kwargs[param_key] = st.select_slider(
                control["label"],
                options=control["options"],
                value=control.get("value", []),
                key=key_name,
                help=control["help"],
            )
        else:
            param_kwargs[param_key] = st.select_slider(
                control["label"],
                options=control["options"],
                value=control.get("value", []),
                key=key_name
            )

    return param_kwargs

def param_controller(param_list: list, para_descriptions: dict, param_controls: dict, desc='',
                     a_copied_dict: bool | str | int =False, expanded:bool=True, cols:int=1, inner_cols:int=1) -> dict:

    params_kwargs = {}

    if cols > 5 or cols < 1:
        cols = 1

    if inner_cols > 5 or inner_cols < 1:
        inner_cols = 1

    m = 0
    param_cols = st.columns([1]*cols)

    for p, param in enumerate(param_list):

        col = p%cols

        with param_cols[col]:

            if param in param_controls:

                with st.expander(
                        f"{desc}\t{chr(9312 + m)}&nbsp;&nbsp;{param}:&nbsp;&nbsp;&nbsp;{para_descriptions.get(param, '')} ",
                        expanded=expanded):

                    params_kwargs[param] = {}
                    st.markdown('---')

                    inner_param_cols = st.columns([1] * inner_cols)
                    getout_inner = False

                    for param_key, control in param_controls[param].items():


                        if not a_copied_dict:
                            key_name = f"{param}_{param_key}"

                        else:

                            if a_copied_dict == 1:

                                key_name = f"{param}_{param_key}_"

                            else:
                                key_name = f"{param}_{param_key}_{a_copied_dict}"


                        if 'inner_col' in control:

                            inner_col = control['inner_col']

                            with inner_param_cols[inner_col]:

                                params_kwargs[param][param_key] = param_control_core(param_key=param_key,control=control,key_name=key_name)[param_key]

                            getout_inner = True

                        else:

                            if getout_inner:

                                st.markdown('---')
                                getout_inner = False

                            params_kwargs[param][param_key] = param_control_core(param_key=param_key, control=control,
                                                                      key_name=key_name)[param_key]


            else:
                with st.expander(
                        f"{desc}\t{chr(9312 + m)}&nbsp;&nbsp;{param}:&nbsp;&nbsp;&nbsp;{para_descriptions.get(param, '')} ",
                        expanded=False):
                    params_kwargs[param] = {}
                    st.markdown('---')

        m += 1

    return params_kwargs


# X-y 对数据生成器
def return_cause_pair(not_pair_data: pd.DataFrame, relation: Literal["AB", "BA", "AB&BA"] = "AB", prefix='', **kwargs):
    if relation == "AB&BA":

        relation = ["AB", "BA"]
    else:
        relation = [relation]

    pair_data = []
    pair_name = []
    pair_cause = []

    for relation_ in relation:

        if relation_ == "AB":
            data_in_pair = [
                pd.concat([pd.DataFrame(not_pair_data.iloc[:, i]), pd.DataFrame(not_pair_data.iloc[:, -1])], axis=1)
                for i in range(not_pair_data.shape[1] - 1)]

        elif relation_ == "BA":

            data_in_pair = [
                pd.concat([pd.DataFrame(not_pair_data.iloc[:, -1]), pd.DataFrame(not_pair_data.iloc[:, i])], axis=1)
                for i in range(not_pair_data.shape[1] - 1)]

        else:
            raise AssertionError(f"return_cause_pair函数无法识别relation参数{relation_}")

        data_in_pair_format = []
        for pair in data_in_pair:
            pair.columns = [0, 1]
            data_in_pair_format.append(pair)

        pair_data.extend(data_in_pair_format)
        pair_name.extend([relation_ + f"_{prefix}[{str(col_name)}]" for col_name in not_pair_data.columns[:-1]])

        if relation_ == "AB":

            pair_cause.extend([1] * len(data_in_pair))


        else:
            pair_cause.extend([0] * len(data_in_pair))

    return pair_data, pair_name, pair_cause




# 实现括号上色的程序1
def colorize_brackets_by_depth(expr: str) -> str:
    """
    对表达式中的括号进行着色，按嵌套层级循环使用不同颜色，
    并正确处理 LaTeX 中指数符号（^）所需的大括号匹配问题。

    参数:
        expr (str): 传入的原始字符串表达式（例如 "y = 2(2(x_2)^2)^2"）

    返回:
        str: 添加 LaTeX 颜色标签后的表达式，可直接用于 st.latex() 渲染
    """
    colors = ['orange', 'red', 'blue', 'yellow', "green", "violet", 'brown', "lime"]
    num_colors = len(colors)
    result = ''  # 最终拼接的 LaTeX 字符串
    depth = 0  # 括号嵌套深度
    stack = []  # 颜色栈，用于匹配每个左括号的颜色
    char_energy = np.array([])  # 存储每个 ^ 所在时的括号层级，用于后续决定在哪里闭合大括号
    char_len = len(expr)

    # 遍历每个字符
    for i, char in enumerate(expr):

        if char == '^':
            # 遇到 ^ 开启指数，追加 ^{ 并记录当前 depth
            result += '^' + '{'
            char_energy = np.append(char_energy, depth)

        elif char == '(':
            # 左括号，根据当前 depth 上色并入栈
            color = colors[depth % num_colors]
            result += rf'\textcolor{{{color}}}{{(}}'
            stack.append(color)
            depth += 1

        elif char == ')':

            # 右括号，先结束 ^ 开启的大括号（若满足闭合条件）

            if stack:
                color = stack.pop()
            else:
                color = colors[0]  # 兜底：括号不匹配时默认颜色

            # 统计在当前 depth 下应该关闭多少个 ^ 所开启的大括号
            energy_exhausted = char_energy >= depth - 1
            char_energy = char_energy[char_energy < depth - 1]

            # 加上当前层级右括号

            result += "}" * np.sum(energy_exhausted)  # 添加闭合括号
            depth -= 1
            result += rf'\textcolor{{{color}}}{{)}}'

        else:
            # 普通字符直接加入结果
            result += char

        # 若到达末尾，补齐所有未关闭的大括号
        if i == char_len - 1:
            result += "}" * len(char_energy)

    return result


def gui_warning(text:str):
    st.markdown(f"""
        <style>
        .custom-warning {{
            text-align: center;
            font-weight: 600;
            font-size: 16px;
            color: #333;
            padding: 18px 18px;
            margin: 10px 0;
            border-radius: 7px;
            border: 2.5px solid #f6c370;
            background-color: transparent;
            width: 100%;
            box-sizing: border-box;
        }}

        @media (prefers-color-scheme: dark) {{
            .custom-warning {{
                color: #ccc;
                border: 2.5px solid #9b8235;
            }}
        }}
        </style>

        <div class="custom-warning">
            {text}
        </div>
    """, unsafe_allow_html=True)


def gui_info(text:str):
    st.markdown(f"""
        <style>
        .custom-info {{
            text-align: center;
            font-weight: 600;
            font-size: 16px;
            color: #333;
            padding: 18px 18px;
            margin: 10px 0;
            border-radius: 7px;
            border: 2.5px solid #244690;
            background-color: transparent;
            width: 100%;
            box-sizing: border-box;
        }}

        @media (prefers-color-scheme: dark) {{
            .custom-info {{
                color: #ccc;
                border: 2.5px solid #4066ca;
            }}
        }}
        </style>

        <div class="custom-info">
            {text}
        </div>
    """, unsafe_allow_html=True)


def gui_success(text:str):
    st.markdown(f"""
        <style>
        .custom-success {{
            text-align: center;
            font-weight: 600;
            font-size: 16px;
            color: #555;
            padding: 18px 18px;
            margin: 10px 0;
            border-radius: 7px;
            border: 2px solid #88c09E;
            background-color: transparent;
            width: 100%;
            box-sizing: border-box;
        }}

        @media (prefers-color-scheme: dark) {{
            .custom-success {{
                color: #ccc;
                border: 2px solid #67AE6E;
            }}
        }}
        </style>

        <div class="custom-success">
            {text}
        </div>
    """, unsafe_allow_html=True)

# 实现括号上色的程序2
def apply_colored_brackets(expr: str) -> str:
    """
    对整个表达式按加号（+）分隔后逐段处理括号着色。

    参数:
        expr (str): 传入的表达式（例如 "y = 2(2(x_2)^2)^2 + sin(πx_1)"）

    返回:
        str: 添加括号颜色的完整表达式
    """
    terms = expr.split('+')
    colored_terms = [colorize_brackets_by_depth(term.strip()) for term in terms]
    return ' + '.join(colored_terms)

def render_dataset_title(title: str, font_size:float|int=26, align="left"):
    # 注入一次全局样式

    st.markdown(f"""
    <style>
    .custom-title-dynamic {{
        font-family: 'Segoe UI Variable Text', 'Roboto', 'Helvetica Neue', sans-serif' !important;
        font-weight: 500;
        text-align: {align} !important;
    }}

    @media (prefers-color-scheme: light) {{
        .custom-title-dynamic {{
            color: #000000 !important;
        }}
    }}

    @media (prefers-color-scheme: dark) {{
        .custom-title-dynamic {{
            color: #ffffff !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

    # 渲染标题，字号使用行内 style
    st.markdown(f"""
    <div class="custom-title-dynamic" style="font-size: {font_size}px;">
        {title}
    </div>
    """, unsafe_allow_html=True)


# 带矩形背景的 detail
def display_detial_dict(d: dict, font_size: int = 16, font_weight: int = 315, margin_bottom: int = 8):
    st.markdown("""
    <style>
    .custom-dict-box {
        display: flex;
        justify-content: space-between;
        font-family: 'Segoe UI Variable Text', 'Roboto', 'Helvetica Neue', sans-serif';
        font-size: 16px;
        font-weight: 315;
        border-radius: 0.2em;
        padding: 0.05em 0.05em;
        border: 5px solid #f0f0f0;
        background-color: #f0f0f0;
        color: black;

    }
    @media (prefers-color-scheme: dark) {
        .custom-dict-box {
            background-color: #121844 !important;
            color: white !important;
            border-color: #121844 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True) # #111638

    # 渲染每个键值对，样式写在class里，动态样式靠CSS控制

    if d:

        for key, value in d.items():
            st.markdown(f"""
                <div class="custom-dict-box" style="
                    font-size: {font_size}px;
                    font-weight: {font_weight};
                    margin-bottom: {margin_bottom}px;
                ">
                    <span>&nbsp;&nbsp;{key}:</span>
                    <span style="font-weight:385">{value}&nbsp;&nbsp;</span>
                </div>
            """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
            <div class="custom-dict-box" style="
                font-size: {font_size}px;
                font-weight: 385;
                margin-bottom: {margin_bottom}px;
            ">
                <span>&nbsp;&nbsp;None</span>
            </div>
        """, unsafe_allow_html=True)


def use_files_download_zip(file_dict:dict, content_type="files_pair"):
    """
    将多个文件打包成 ZIP 并返回字节对象。

    参数:
        file_dict (dict): {文件名: 文件字节内容}
        zip_name (str): ZIP 文件名（可选，默认 "download.zip"）

    返回:
        bytes: ZIP 文件的字节内容
    """

    if file_dict:

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

            for data_name, features in file_dict.items():

                for feature_name, df in features[f'{content_type}'].items():

                    # 转成 CSV（不保存到磁盘，直接写入内存）
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    # 文件路径为: data_name/feature_name.csv
                    zf.writestr(f"{data_name}/{feature_name}.csv", csv_bytes)

        # 将指针移动到开头
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    else:

        return None



# 带下划线的副标题
def render_section_title(
        text: str,
        align="left",
        font_size=20,
        line_color='#3580f5',
        underline: bool = True
):
    """
    渲染一个美观的标题组件，可选是否带下划线，并自动适配亮/暗色主题。

    参数：
    - text: 标题文本
    - align: 对齐方式，'left' 或 'center'
    - font_size: 字体大小（单位：px）
    - line_color: 下划线颜色（如启用下划线）
    - underline: 是否显示下划线（默认 True）
    """
    alignment_css = "margin: 0 auto;" if align == "center" else "margin-left: 7px;"
    underline_css = f"border-bottom: 4.2px solid {line_color}; border-radius: 1.5px;" if underline else ""

    st.markdown(f"""
        <style>
        .section-title {{
            font-size: {font_size}px;
            font-weight: 700;
            text-align: {align};
            padding-bottom: 6px;
            display: block;
            width: fit-content;
            {alignment_css}
            {underline_css}
            color: #ffffff;
        }}

        @media (prefers-color-scheme: light) {{
            .section-title {{
                color: #000000;
            }}
        }}
        </style>

        <div class="section-title">{text}</div>
    """, unsafe_allow_html=True)


def render_noline_title(
        text: str,
        font_size=17,
        line_color='#3580f5',
        underline: bool = True
):
    """
    渲染一个美观的标题组件，可选是否带下划线，并自动适配亮/暗色主题。

    参数：
    - text: 标题文本
    - align: 对齐方式，'left' 或 'center'
    - font_size: 字体大小（单位：px）
    - line_color: 下划线颜色（如启用下划线）
    - underline: 是否显示下划线（默认 True）
    """

    st.markdown(f"""
        <style>
        .section-noline_title {{
            font-size: {font_size}px;
            font-weight: 500;
            text-align: left;
            padding-bottom: 0px;
            display: block;
            width: fit-content;
            color: #e0e0e0;
            margin-top: 7px;
        }}

        @media (prefers-color-scheme: light) {{
            .section-noline_title {{
                color: #404040;
            }}
        }}
        </style>

        <div class="section-noline_title">{text}</div>
    """, unsafe_allow_html=True)



# 网页副分割线
def hr_second(height=2, dark_color="#ffffff", light_color="#000000", left_px=0, right_px=0):

    """
    渲染一条自定义分隔线，自动适配暗/亮模式。

    参数:
    - height: 线条高度（单位：px）
    - dark_color: 暗色模式下线条颜色（默认白色）
    - light_color: 浅色模式下线条颜色（默认黑色）
    """

    st.markdown(
        f"""
        <style>
        .custom-hr {{
            border: none !important;
            height: {height}px !important;
            background-color: {dark_color} !important;
            margin: 1rem 0;
            margin-left: {left_px}px !important;
            margin-right: {right_px}px !important;
        }}

        @media (prefers-color-scheme: light) {{
            .custom-hr {{
                background-color: {light_color} !important;
            }}
        }}
        </style>

        <hr class="custom-hr">
        """,
        unsafe_allow_html=True
    )




script_dir = os.path.dirname(os.path.abspath(__file__))
dpls_dir = os.path.join(script_dir, 'dpls_scripts')


if os.path.exists(dpls_dir):
    pass
else:
    os.makedirs(dpls_dir)


dpls_files = [
    f for f in os.listdir(dpls_dir)
    if f.endswith('.py')
]

dpls_modules = [f[:-3] for f in dpls_files]


selected_dpls_version = st.selectbox('选择 DPLS 版本', dpls_modules, key='selected_dpls_version')

# 动态导入模块
if "dpls_version" not in st.session_state or st.session_state.get("dpls_version") != selected_dpls_version:
    st.session_state.dpls_version = selected_dpls_version
    module_name = f"dpls_scripts.{selected_dpls_version}"  # 注意这里没有 .py
    DPLS_module = importlib.import_module(module_name)
    st.session_state.DPLS_module = DPLS_module
else:
    DPLS_module = st.session_state.DPLS_module

# 假设每个模块里有 DPLS 类或对象
st.session_state.DPLS_version = DPLS_module.DPLS
DPLS = st.session_state.DPLS_version


def cal_DPLS_pred(file_value:pd.DataFrame,  **kwargs):

    value_copy = file_value.copy()

    DPLS_obj = DPLS(**kwargs).fit(value_copy[[kwargs['reason']]], value_copy[[kwargs['result']]], **kwargs)

    return DPLS_obj.R2[0], DPLS_obj.y_pred[0]


def single_workspace(raw: dict, total_file, thread=1, expand=True, print_pred=False, block_id="test",
                     checking_file:str | None = None, description_type:Literal['latex', 'str'] = 'str'):


    check_file_panel = st.columns([1.35, 3])

    with check_file_panel[1]:

        check_file_panel_2_expander = st.expander("File Details", expanded=expand)


    def a_single_workspace(db_values):

        if print_pred:

            if "files_dpls_obj" in db_values:
                checking_dpls_objs: dict = db_values["files_dpls_obj"]

            else:
                checking_dpls_objs: dict = parallel_wrapper(cal_DPLS_obj, db_values["files_pair"],
                                                            desc="cal_DPLS_obj", reason=0, result=1,
                                                            thread=thread)

                db_values["files_dpls_obj"] = checking_dpls_objs

        def click_and_show_():

            check_plot_col, check_data_info_col = st.columns([2, 1])

            with check_data_info_col:
                with st.expander(f'**{values.shape}**', expanded=True):
                    st.dataframe(values, height=382)

            with check_plot_col:

                fig_db = go.Figure()

                # 添加 y_exp（蓝色），优先添加以保证 preds 最上层

                fig_db.add_scatter(
                    x=values[0],
                    y=values[1],
                    mode='markers',
                    name='y_obs',
                    marker=dict(
                        color='#3580F5',
                        size=6,
                        opacity=0.75
                    )
                )

                if print_pred:
                    checking_dpls_obj = checking_dpls_objs.get(name, list(checking_dpls_objs.values())[db_values_count])
                    checking_pred = checking_dpls_obj.y_pred[0]

                    if checking_pred is not None:

                        fig_db.add_scatter(
                            x=values[0],
                            y=checking_pred.flatten(),
                            mode='markers',
                            name='preds',
                            marker=dict(
                                color='#FFB420',
                                size=4.5,
                                opacity=0.82
                            )
                        )

                # 更新坐标轴标签
                fig_db.update_layout(
                    xaxis_title=name[0],
                    yaxis_title=name[1],
                    legend=dict(
                        traceorder="normal"  # 图例顺序按添加顺序排列
                    )
                )

                # 更新布局
                fig_db.update_layout(
                    title=dict(
                        text=f"",
                        x=0.55,  # 居中，可调
                        xanchor='right',
                        font=dict(size=19),
                    ),
                    showlegend=False,
                    height=430,
                    margin=dict(t=40, b=10, l=7, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, showline=False, ticks='',
                               showticklabels=True),
                    yaxis=dict(showgrid=False, zeroline=False, showline=False, ticks='',
                               showticklabels=True)
                )

                st.plotly_chart(fig_db, use_container_width=True)
                st.markdown("")

        first_expand = True
        db_values_count = 0

        for name, values in db_values["files_pair"].items():

            if "X_name" in db_values:
                expander_name = db_values["X_name"].get(name, list(db_values["X_name"].values())[db_values_count])
            else:
                expander_name = name

            if first_expand:
                with st.expander(f'{expander_name}', expanded=first_expand):
                    click_and_show_()

                after_expand = st.expander("**⋯**")
                first_expand = False

            else:

                with after_expand:
                    with st.expander(f'{expander_name}', expanded=True):
                        click_and_show_()

            db_values_count += 1

    with check_file_panel[0]:

        with st.expander("", expanded=expand):
            # 文件列表标题
            raw_x_cols = st.columns([3, 4])
            raw_x_cols[0].markdown(
                "<div style='text-align: left; margin-top: 30px; padding-left:10px;'>文件名</div>",
                unsafe_allow_html=True)
            raw_x_cols[1].markdown(
                f"<div style='text-align: right; margin-top: 23px; padding-right: 20px;'> <span style='color: #55dd99; font-size:20px;'><strong>{total_file}</strong></span> files</div>",
                unsafe_allow_html=True)

            st.markdown("---")

            for db_name, db_values in raw.items():

                if st.button(f'{db_name}', key=f"{block_id}-{db_name}-button", use_container_width=True):

                    checking_file = db_name


            if not checking_file:

                checking_file = db_name


            with check_file_panel_2_expander:

                if raw[checking_file].get('description', False):

                    db_description = raw[checking_file]['description']

                else:

                    db_description = "No description"

                if description_type == 'latex':

                    latex_expr = apply_colored_brackets(db_description)
                    st.latex(latex_expr)

                else:

                    raw_y_cols = st.columns([2, 6.6])

                    raw_y_cols[1].markdown(
                        f"<div style='text-align: right; margin-top: 20px; padding-right: 20px; font-weight:400;'>{db_description}</div>",
                        unsafe_allow_html=True)

                    raw_y_cols[0].markdown(
                        f"<div style='text-align: left; margin-top: 20px; padding-left: 20px;'> <strong>[{checking_file}] </strong></div>",
                        unsafe_allow_html=True)

                st.markdown("---")

                a_single_workspace(raw[checking_file])

            return checking_file


def fusion_workspace(raw, block_id, thread, total_file, checking_file:str | None = None, description_type:Literal['latex', 'str'] = 'str', ):


    check_file_panel = st.columns([1.35, 3])

    with check_file_panel[1]:

        check_file_panel_2_expander = st.expander("File Details", expanded=True)


    def a_fusion_workspace(X, y, checking=None):

        if X is None or y is None:

            return 0

        print("a_fusion_workspace_SHAPE", X.shape, y.shape)

        # 使用 expander 创建一个可折叠/展开的区域
        DPLSR_param_dict_ = DPLSR_param_dict.copy()

        DPLS_kwargs = param_controller(
            param_list=['DPLSR'],
            para_descriptions={'DPLSR': method_descriptions.get('DPLSR')},
            param_controls={"DPLSR": DPLSR_param_dict_},
            desc='方法',
            a_copied_dict=f"{block_id}", expanded=False,
        )
        if not DPLS_kwargs['DPLSR']["distance_pattern"]:
            DPLS_kwargs['DPLSR']["distance_pattern"] = ['Euc']

        DPLS_kwargs['checking'] = checking

        plot_expander = st.expander('fusion_workspace', expanded=True)

        with plot_expander:

            plot_sep_0, plot_P_col, plot_sep_1, plot_sep_3, plot_DPLSR_col = st.columns(
                [.01, 4, .2, 1.6, 2])

            st.markdown("---")

        x_picked_eg = list(X.columns)

        with plot_P_col:

            enforce_P_col, x_num_col = st.columns([1, 1])

            with enforce_P_col:

                enforce_P = st.slider("硬定位的 P: ", min_value=-1, max_value=DPLS_kwargs['DPLSR']['max_iter'] - 1,
                                      value=-1, key=f'{block_id}_enforce_P')

            with x_num_col:

                x_selected = st.multiselect("使用的 X: ", options=['All'] + x_picked_eg, key=f'{block_id}_x_num',
                                            default=['All'])

                if f"All" in x_selected:
                    x_selected = x_picked_eg

                if not x_selected:
                    x_selected = x_picked_eg

                DPLS_kwargs['x_num'] = x_selected

                x_eg_use = copy.deepcopy(X)[x_selected]

        with st.spinner("正在拟合..."):
            # 检测 dpls 参数是否被改动过
            if f"{block_id}_eg_created" not in st.session_state:

                st.session_state[f"{block_id}_dpls_kwargs"] = DPLS_kwargs
                pred_obj = DPLS(**DPLS_kwargs["DPLSR"]).fit(x_eg_use[x_selected].copy(), y,
                                                            **DPLS_kwargs["DPLSR"])
                st.session_state[f'{block_id}_pred_obj'] = pred_obj
                dpls_change = False

            else:

                dpls_change = (st.session_state.get(f"{block_id}_dpls_kwargs", None) != DPLS_kwargs)
                st.session_state[f"{block_id}_dpls_kwargs"] = DPLS_kwargs

                if dpls_change:

                    pred_obj = DPLS(**DPLS_kwargs["DPLSR"]).fit(x_eg_use[x_selected].copy(), y,
                                                                **DPLS_kwargs["DPLSR"])
                    st.session_state[f'{block_id}_pred_obj'] = pred_obj

                else:
                    pred_obj = st.session_state[f'{block_id}_pred_obj']

            st.session_state[f'{block_id}_eg_created'] = True

        with plot_sep_3:
            if DPLS_kwargs["DPLSR"]["R_mode"] == 'single':
                pass
            else:

                st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <h3 style='text-align: right; font-size: 28px; font-weight: bold;'>
                        P: 
                        <span style='color: #3580f5; font-size: 30px; font-weight: bold;'>
                            {pred_obj.p[0] if enforce_P == -1 else enforce_P}
                        </span>
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

        with plot_DPLSR_col:
            if DPLS_kwargs["DPLSR"]["R_mode"] == 'single':
                pass
            else:
                st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <h3 style='text-align: right; font-size: 28px; font-weight: bold;'>
                        DPLSR: 
                        <span style='color: #3580f5; font-size: 30px; font-weight: bold;'>
                            {pred_obj.R2[0] if enforce_P == -1 else pred_obj.y_pred_R2[0][enforce_P] :.2f}
                        </span>
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

        # 定义两列

        with plot_expander:

            cols = st.columns(2)

        fig_dict = {}
        # 创建一个空图

        for idx, x_i in enumerate(x_selected):

            x = list(x_eg_use.columns).index(x_i)

            if enforce_P == -1:

                if DPLS_kwargs["DPLSR"]["R_mode"] == 'single':

                    y_pred = pred_obj.y_pred[x]

                    R = pred_obj.R2[x]
                    P = pred_obj.p[x]

                else:

                    y_pred = pred_obj.y_pred[0]

                    R = pred_obj.R2[0]
                    P = pred_obj.p[0]

            else:

                if DPLS_kwargs["DPLSR"]["R_mode"] == 'single':

                    y_pred = pred_obj.y_preds[x][:, enforce_P]
                    R = pred_obj.y_pred_R2[x][enforce_P]

                else:

                    y_pred = pred_obj.y_preds[0][:, enforce_P]
                    R = pred_obj.y_pred_R2[0][enforce_P]

                P = enforce_P

            x_eg_use_i = pd.DataFrame()
            x_eg_use_i[x_i] = pred_obj.X.copy()[:, x]

            x_eg_use_i['y'] = y
            x_eg_use_i['preds'] = y_pred  # 使用完整预测值

            fig = go.Figure()
            # 添加 y_exp（蓝色），优先添加以保证 preds 最上层

            fig.add_scatter(
                x=x_eg_use_i[x_i],
                y=x_eg_use_i['y'],
                mode='markers',
                name='y_exp',
                marker=dict(
                    color='#3580F5',
                    size=6,
                    opacity=0.75
                )
            )

            # 最后添加 preds（橙色），确保在最上层
            fig.add_scatter(
                x=x_eg_use_i[x_i],
                y=x_eg_use_i['preds'],
                mode='markers',
                name='preds',
                marker=dict(
                    color='#FFB420',
                    size=4.5,
                    opacity=0.82
                )
            )

            # 更新坐标轴标签
            fig.update_layout(
                xaxis_title=x_i,
                yaxis_title=f'y',
                legend=dict(
                    traceorder="normal"  # 图例顺序按添加顺序排列
                )
            )

            # 更新布局
            fig.update_layout(
                title=dict(
                    text=f"[x_{x_selected.index(x_i) + 1}] [P: {P}] [R: {R:.2f}]",
                    x=0.55,  # 居中，可调
                    xanchor='right',
                    font=dict(size=19),
                ),
                showlegend=False,
                height=430,
                margin=dict(t=40, b=10, l=7, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showline=False, ticks='', showticklabels=True),
                yaxis=dict(showgrid=False, zeroline=False, showline=False, ticks='', showticklabels=True)
            )

            with cols[idx % 2]:
                st.plotly_chart(fig, use_container_width=True, key=f"{block_id}_{x_i}create_plotly_chart")

            fig_dict[f"plots_{x_i}"] = fig

        create_check_file_expander = st.expander("检视预览函数样本", expanded=True)
        with create_check_file_expander:
            formula_x_col, formula_check_file_sep, formula_y_col = st.columns([.738, 0.025, .382])

            with formula_x_col:
                st.markdown("")
                st.markdown("")
                render_section_title("来自数据的自变量:")

                with st.expander(f"**{X.shape}**", expanded=False):
                    st.markdown("---")
                    st.dataframe(x_eg_use.copy(), height=795, hide_index=True)

            with formula_y_col:
                st.markdown("")
                st.markdown("")

                render_section_title("因变量")
                with st.expander(f"**{y.shape}**",
                                 expanded=False):
                    st.markdown("---")
                    st.dataframe(y, height=795, hide_index=True)


    with check_file_panel[0]:

        with st.expander("", expanded=True):

            # 文件列表标题
            raw_x_cols = st.columns([3, 4])
            raw_x_cols[0].markdown(
                "<div style='text-align: left; margin-top: 30px; padding-left:10px;'>文件名</div>",
                unsafe_allow_html=True)
            raw_x_cols[1].markdown(
                f"<div style='text-align: right; margin-top: 23px; padding-right: 20px;'> <span style='color: #55dd99; font-size:20px;'><strong>{total_file}</strong></span> files</div>",
                unsafe_allow_html=True)

            st.markdown("---")

            for db_name, db_values in raw.items():

                if st.button(f'{db_name}', key=f"{block_id}-{db_name}-button", use_container_width=True):

                    checking_file = db_name


            if not checking_file:

                checking_file = db_name


            with check_file_panel_2_expander:

                if raw[checking_file].get('description', False):

                    db_description = raw[checking_file]['description']

                else:

                    db_description = "No description"

                if description_type == 'latex':

                    latex_expr = apply_colored_brackets(db_description)
                    st.latex(latex_expr)

                else:

                    raw_y_cols = st.columns([2, 6.6])

                    raw_y_cols[1].markdown(
                        f"<div style='text-align: right; margin-top: 20px; padding-right: 20px; font-weight:400;'>{db_description}</div>",
                        unsafe_allow_html=True)

                    raw_y_cols[0].markdown(
                        f"<div style='text-align: left; margin-top: 20px; padding-left: 20px;'> <strong>[{checking_file}] </strong></div>",
                        unsafe_allow_html=True)

                st.markdown("---")

                a_fusion_workspace(X = raw[checking_file].get("X", None), y = raw[checking_file].get("y", None), checking=checking_file)

            print("checking_file", checking_file)

            return checking_file


