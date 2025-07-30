# 4-1 方法的描述


method_descriptions = {

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


# 4-2 方法参数映射（用于动态显示对应的子参数控件）
DPLSR_param_dict = {
    "cv": {
        "type": "slider",
        "label": "多重检验折数",
        "min": 1,
        "max": 10,
        "step": 1,
        "value": 1
    },
    "max_iter": {
        "type": "slider",
        "label": "DPLS最大迭代层",
        "min": 1,
        "max": 500,
        "value": 20
    },

    "power": {
        "type": "slider",
        "label": "距离矩阵自乘幂",
        "min": 0,
        "max": 10,
        "value": 0,
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
        "value": "Fit_rectify"
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
PATH_param_dict = {
    "Path_centre": {"type": "checkbox", "label": "中心化后再求 PATH", "value": False},
    "Path_normal": {"type": "checkbox", "label": "PATH 除以输入样本的标准差", "value": False},
    "Path_window": {"type": "slider", "label": "delta间隔", "min": 1, "max": 10, "value": 1},
    "Sort_by": {"type": "selectbox",
                "label": "排序模式", "help":"[reason]: 仅根据 reason 排序,重复值保留原始顺序, [all]: result 参与排序,作为排序的次级依据",
                "options": ["all", "reason", ], "value": "all"},
}


method_param_controls = {

    "DPLSR": {

         "need_P": {
             "type": "checkbox",
             "label": "返回选择的P",
             "value": False,
         },

         "need_Rs":{
             "type": "checkbox",
             "label": "返回每个P的R2",
             "value": False,
         },
             } | DPLSR_param_dict,

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

