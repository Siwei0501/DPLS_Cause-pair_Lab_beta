import copy
import glob
import importlib.util
import io
import json
import os
import sys
import time
import traceback
import webbrowser
import zipfile
from typing import Callable
from typing import Literal, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import Parallel, delayed
from stqdm import stqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# 导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cause_pair_functions.casual_pair_tester import process, algorithms, return_values_DF
from cause_pair_functions.muti_func_test import gen_y_exp
from cause_pair_functions.DPLS_jj import DPLS

# 0 设置与定义------------------------------------------------------------------------------------------------------------

st.markdown("""
<style>
/* 将整个内容区域整体往上移动 65px */
.block-container {
    position: relative;
    top: -65px !important;
}
</style>
""", unsafe_allow_html=True)


# 0-2 设置路径 -----------------------------------------------------------------------------------------------------------


# 脚本路径
script_dir = os.path.dirname(os.path.abspath(__file__))
#本地数据路径
local_data_dir = os.path.join(script_dir, 'Cause_DBs')
#配置路径
config_dir = os.path.join(script_dir, 'config_file')
#版本号
now_version = "beta4"

# 全局功能函数定义区 -------------------------------------------------------------------------------------------------------

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
def gen_seed(param_num: int, rand_seed: int, gen_times: int=1):
    np.random.seed(rand_seed)
    seed_list = np.random.randint(0, 50000, [gen_times, param_num]).reshape(gen_times, param_num)
    np.random.seed(None)
    return seed_list.tolist()


# X-y 对数据生成器
def return_cause_pair(not_pair_data:pd.DataFrame, relation:Literal["AB", "BA", "AB&BA"]="AB",prefix='', **kwargs):

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
        pair_name.extend([relation_ + f"{prefix}[{str(col_name)}]" for col_name in not_pair_data.columns[:-1]])

        if relation_ == "AB":

            pair_cause.extend([1]*len(data_in_pair))


        else:
            pair_cause.extend([0]*len(data_in_pair))

    return pair_data, pair_name, pair_cause

def cal_DPLS_pred(file_value:pd.DataFrame,  **kwargs):

    value_copy = file_value.copy()

    DPLS_obj = DPLS(**kwargs).fit(value_copy[[kwargs['reason']]], value_copy[[kwargs['result']]], **kwargs)

    return DPLS_obj.R2[0], DPLS_obj.y_pred[0]

# 读取文件描述
def read_txt_or_default(filepath):
    description_path = os.path.join(filepath, 'description.txt')

    try:
        with open(description_path, 'r', encoding='utf-8') as f:
            return f.read()
    except (FileNotFoundError, PermissionError) as e:
        return "No description"

# 参数收集器
def param_controller(param_list: list, para_descriptions: dict, param_controls: dict, desc='', a_copied_dict: bool | str | int =False, expanded=True) -> dict:
    param_kwargs = {}

    m = 0
    for param in param_list:

        if param in param_controls:

            with st.expander(
                    f"{desc}\t{chr(9312 + m)}&nbsp;&nbsp;{param}:&nbsp;&nbsp;&nbsp;{para_descriptions.get(param, '')} ",
                    expanded=expanded):
                param_kwargs[param] = {}
                st.markdown('---')

                for param_key, control in param_controls[param].items():

                    if not a_copied_dict:
                        key_name = f"{param}_{param_key}"

                    else:

                        if a_copied_dict == 1:

                            key_name = f"{param}_{param_key}_"

                        else:
                            key_name = f"{param}_{param_key}_{a_copied_dict}"

                    if control["type"] == "checkbox":

                        if "help" in control:

                            param_kwargs[param][param_key] = st.checkbox(control["label"],
                                                                         value=control.get("value", False),
                                                                         key=key_name,
                                                                         help=control["help"]
                                                                         )

                        else:

                            param_kwargs[param][param_key] = st.checkbox(control["label"],
                                                                         value=control.get("value", False),
                                                                         key=key_name,
                                                                         )


                    elif control["type"] == "slider":

                        if "help" in control:

                            param_kwargs[param][param_key] = st.slider(
                                control["label"],
                                min_value=control["min"],
                                max_value=control["max"],
                                value=control["value"],
                                step=control.get("step", 1),
                                key=key_name,
                                help=control["help"],
                            )

                        else:

                            param_kwargs[param][param_key] = st.slider(
                                control["label"],
                                min_value=control["min"],
                                max_value=control["max"],
                                value=control["value"],
                                step=control.get("step", 1),
                                key=key_name
                            )

                    elif control["type"] == "selectbox":

                        if "help" in control:

                            param_kwargs[param][param_key] = st.selectbox(
                                control["label"],
                                control["options"],
                                index=control["options"].index(control["value"]),
                                key=key_name,
                                help=control["help"],
                            )
                        else:

                            param_kwargs[param][param_key] = st.selectbox(
                                control["label"],
                                control["options"],
                                index=control["options"].index(control["value"]),
                                key=key_name
                            )

                    elif control["type"] == "multiselect":

                        if "help" in control:

                            param_kwargs[param][param_key] = st.multiselect(
                                control["label"],
                                options=control["options"],
                                default=control.get("value", []),
                                key=key_name,
                                help=control["help"],
                            )
                        else:
                            param_kwargs[param][param_key] = st.multiselect(
                                control["label"],
                                options=control["options"],
                                default=control.get("value", []),
                                key=key_name
                            )

                    elif control["type"] == "select_slider":

                        if "help" in control:

                            param_kwargs[param][param_key] = st.select_slider(
                                control["label"],
                                options=control["options"],
                                value=control.get("value", []),
                                key=key_name,
                                help=control["help"],
                            )
                        else:
                            param_kwargs[param][param_key] = st.select_slider(
                                control["label"],
                                options=control["options"],
                                value=control.get("value", []),
                                key=key_name
                            )



        else:
            with st.expander(
                    f"{desc}\t{chr(9312 + m)}&nbsp;&nbsp;{param}:&nbsp;&nbsp;&nbsp;{para_descriptions.get(param, '')} ",
                    expanded=False):
                param_kwargs[param] = {}
                st.markdown('---')

        m += 1
    return param_kwargs


# 本地文件读取器
def data_presenter(dataset_name, **dataset_kwargs):

    module_name = "pair_data_presenter"
    module_path = os.path.join(local_data_dir, dataset_name, 'pair_data_presenter.py')

    spec = importlib.util.spec_from_file_location(module_name, module_path)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # 将模块添加到 sys.modules
    spec.loader.exec_module(module)  # 执行模块代码

    # 现在可以调用模块中的内容
    pair_presented_data, col_names, cause_y = module.return_cause_pair(**dataset_kwargs)  # 假设模块中有 some_function()

    return pair_presented_data, col_names, cause_y


# 网页外观套件 -----------------------------------------------------------------------------------------------------------


# 细字体副标题
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

def gray_box(text="这是默认内容", color="#f0f0f0", font_size="1rem"):
    html = f"""
    <div style="
        background-color: {color};
        padding: 1em 1.2em;
        border-radius: 0.5em;
        border: 1px solid #ddd;
        color: black;
        font-size: {font_size};
        line-height: 1.5;

    ">
        {text}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)



# 模拟数据生成函数定义区 ----------------------------------------------------------------------------------------------------


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
    "指数幂函数": lambda f: 2 ** (5 * (f + 1)),
    "高频正弦函数": lambda f: np.sin(2 * np.pi * f),
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




# 0-1 标题 --------------------------------------------------------------------------------------------------------------


# 网页名字
st.set_page_config(page_title="DPLS Cause-pair Lab", layout="wide")

# 网页主标题及 Cause 动画
st.markdown("""
<style>
/* 渐变动画关键帧 */
@keyframes animated-gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 标题容器整体上移 */
.custom-title-container {
    position: relative;
    top: -23px;
    left: 35px;
    margin-bottom: 10px;
    margin-top: 25px;
}

/* 主标题样式 */
.custom-title {
    text-align: right;
    font-size: 3.1em !important;
    font-family: "Segoe UI Variable Text", "Roboto", "Helvetica Neue", sans-serif !important;
    margin-left: 50px !important;
    padding: 0;
}

/* Cause 渐变文字样式 */
.gradient-text {
    background: linear-gradient(135deg, #99e3f4, #6382f4, #9864f2);
    background-size: 300% 300%;
    background-position: 0% 50%;
    animation: animated-gradient 7s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
}

/* beta 标签 */
.beta-tag {
    font-size: 0.4em;
    vertical-align: top;
    margin-left: -9px;
}
</style>
""", unsafe_allow_html=True)

# 构建 HTML 标题
title_html = f"""
<div class="custom-title-container">
    <h1 class="custom-title">
        DPLS&nbsp;<span class="gradient-text">Cause-pair&nbsp;</span>Laboratory
        <span class="beta-tag">{now_version}</span>
    </h1>
</div>

"""
title_bar, title_name = st.columns([0.65,1])


# 0-1-1 标题主进度条 ------------------------------------------------------------------------------------------------------

with title_bar:

    st.empty()

    title_bar_progress = None
    total_mission=1
    now_mission=0

with title_name:
    st.markdown(title_html, unsafe_allow_html=True)


def parallel_wrapper(
        func: Callable,
        file_value_dict: dict,
        desc: str,
        thread: int = 1,
        **kwargs

) -> dict:

    global title_bar_progress

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


# 0-2-1 自定义渐变线 ------------------------------------------------------------------------------------------------------


st.markdown("""
<style>


@media (prefers-color-scheme: dark) {
    .custom-mid-hr {
        margin: 1px 0 !important;
        border: 0;
        position: relative;
        top: 0px;
        left: 0px;
        height: 0px !important;  /* 你可以调粗细 */
        background: #999 !important;

    }
}

@media (prefers-color-scheme: light) {
    .custom-mid-hr {
        margin: 1px 0 !important;
        border: 0;
        position: relative;
        top: 0px;
        left: 0px;
        height: 0px !important;
        background: #eee !important;

    }
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>


@media (prefers-color-scheme: dark) {
    .custom-title-hr {
        margin: 2px 0 !important;
        border: 0;
        position: relative;
        top: -40px;
        left: 0px;
        height: 2px !important;  /* 你可以调粗细 */
        background: #333 !important;
        background-image: linear-gradient(to right, #878796, #2545df, #8060ff) !important;

    }
}

@media (prefers-color-scheme: light) {
    .custom-title-hr {
        margin: 2px 0 !important;
        border: 0;
        position: relative;
        top: -40px;
        left: 0px;
        height: 2.2px !important;
        background: #eee !important;
        background-image: linear-gradient(to right, #bbb, #25f, #74c) !important;

    }
}

</style>
""", unsafe_allow_html=True)

# 渲染装饰线
st.markdown("<hr class='custom-title-hr'>", unsafe_allow_html=True)


# 0-3 背景色定义----------------------------------------------------------------------------------------------------------


st.markdown("""
    <style>

    /* 主页面背景：根据系统主题切换 */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(8deg, #070911, #181826);
            color: #e6e6e6;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(125deg, #212127, #242730, #151821);
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }

    }

    @media (prefers-color-scheme: light) {
        .stApp {
            background: linear-gradient(160deg, #f9f9f9, #f9f9f9, #f9f9f9);
            color: #000000;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(70deg, #dfdfdf, #e9e9e9, #e0e0e0);
        }
        section[data-testid="stSidebar"] * {
            color: black !important;
        }
    }


                /* 暗色主题下按钮样式 */
        @media (prefers-color-scheme: dark) {
            .stButton > button {
                background-color: #1545cc !important;
                color: white !important;
                border: none !important;
                outline: none !important;
                transition: background 0.7s linear;
            }

            .stButton > button:hover {
                background: linear-gradient(300deg, #2262f1, #1852d3,  #1545cc, #1545cc, #1852d3, #2262f1);
                background-size: 400% 100%;
                animation: button-gradient-move-dark 3.6s linear infinite;
            }

            .stButton > button:active {
                background-color: #334455 !important;
                animation: none !important;
            }
        }

        /* 浅色主题下按钮样式 */
        @media (prefers-color-scheme: light) {
            .stButton > button {
                background-color: #ffffff !important;
                color: black !important;
                border: 1px solid #888888 !important;
                outline: none !important;
                transition: background 0.1s linear;
            }

            .stButton > button:hover {
                background: white !important;
                background-size: 200% 100%;
                border: 2.5px solid #3580f5 !important;
            }

            .stButton > button:active {
                background-color: #ffffff !important;
                border: 2.5px solid #2660d3 !important;
                outline: none !important;

            }
        }

        /* 渐变动画定义 */
        @keyframes button-gradient-move-dark {
            0% { background-position: 0% 50%; }
            100% { background-position: -400% 50%; }

        }

            @keyframes button-gradient-move-light {
        0% { background-position: 0% 50%; }
        100% { background-position: -100% 50%; }

        }

    /* 成功提示色调 */
    .stAlert[data-baseweb="notification"][data-kind="success"] {
        background-color: #e6f2ff;
        color: #1f6391;
    }


    /* Streamlit Dataframe 居中对齐 */
    .stDataFrame div[data-testid="stVerticalBlock"] td {
        text-align: center !important;
    }
    .stDataFrame th {
        text-align: center !important;
    }


    </style>
""", unsafe_allow_html=True)


# 1 定义主内容区 ---------------------------------------------------------------------------------------------------------


host_panel_detail, host_panel_sep_2, host_panel_setting, host_panel_sep3, host_config_panel= st.columns(
    [.666, 0.013, .185, 0.007, .135])

# host_panel_detail: 详细信息区
# host_panel_setting: 样本设置区
# host_config_panel: 配置区


# 详细信息内容区
with host_panel_detail:
    panel_detail_title, panel_detail_2 = st.columns([3, 1])
    with panel_detail_title:
        # render_dataset_title('详细信息', )
        st.subheader("详细信息")
    st.markdown("---")


# 1-1 配置区 ----------------------------------------------------------------------------------------------------------------


config_input = False
now_config_name = None
confirm_deleted=False



with host_config_panel:

    # render_dataset_title('配置',)
    st.subheader("配置", help="什么是 [配置]? 配置可以理解为实验的 [配方], 存储 [配置] 可以让你轻松地 [复现] 以前的实验 🤓")

    st.markdown('---')

    import os

    # 1-1-1 读取配置 ----------------------------------------------------------------------------------------------------

    config_files = [
        f for f in os.listdir(config_dir)
        if f.endswith('.json')
    ]

    # 1-1-2 按修改时间从新到旧排序----------------------------------------------------------------------------------------------------

    config_panel_expander = st.expander("**DPLS_lab**", expanded=True)

    with config_panel_expander:
        st.markdown('')
        st.markdown('')
        st.markdown("---")



        config_files = sorted(
            config_files,
            key=lambda f: os.path.getmtime(os.path.join(config_dir, f)),
            reverse=True
        )

        selected_config = st.selectbox('选择运行配置文件', config_files)

        load_config_, refresh_config = st.columns([1,.25])

        # 1-1-3 载入配置功能 ----------------------------------------------------------------------------------------------------

        with load_config_:

            load_config_button = st.button('载入配置', use_container_width=True)
            load_last_config = st.button('运行上次配置', use_container_width=True)
            if load_last_config:
                if os.path.exists(os.path.join(config_dir, 'last_run_config.json')):
                    with open(os.path.join(config_dir, 'last_run_config.json'), "r", encoding="utf-8") as f:
                        last_config = json.load(f)

                    if "now_config_name" in st.session_state:
                        del st.session_state["now_config_name"]

                    st.session_state.update(last_config)
                    config_input = True

                    st.success("✔ 已成功载入上次运行配置")
                    st.rerun()

                else:
                    st.warning("⚠️ 没有找到上次运行配置")

        # 1-1-4 刷新与文件浏览器按钮 ----------------------------------------------------------------------------------------------------

        with refresh_config:
            refresh_config_button = st.button('**↻**', use_container_width=True)

            if refresh_config_button:

                st.rerun()

            manage_config_button = st.button("📁", use_container_width=True)

            if manage_config_button:
                webbrowser.open_new_tab(config_dir)

        # 1-1-5 加载配置功能 ------------------------------------------------------------------------------------------------------------------------------

        if load_config_button:

            if len(config_files) > 0:
                with open(os.path.join(config_dir, selected_config), 'r', encoding='utf-8') as f:
                    input_config = json.load(f)
                    config_input = True
                    st.session_state.update(input_config)
                    st.rerun()

                    # st.info(f'配置已载入')

                # cancel_config_button = st.button('取消配置')
                # if cancel_config_button:
                #     for key in list(st.session_state.keys()):
                #         del st.session_state[key]
                #     st.rerun()

        # 1-1-6 存储配置功能 -----------------------------------------------------------------------------------------------------

        st.markdown("---")

        now_config_name = st.text_input('本次配置文件名', key="now_config_name", placeholder='在此输入文件名', help='输入文件名即可, 不需要加后缀如.json等')
        save_config_button = st.button('存储配置', use_container_width=True, icon="💾")
        output_config_button_ = st.button("导出配置", use_container_width=True)

        st.markdown('---')

        # 1-1-7 删除配置功能 ----------------------------------------------------------------------------------------------------

        delete_config_button = st.button("删除所有配置", use_container_width=True)

        if delete_config_button:

            st.warning("⚠️ 确定要删除数据吗？此操作无法撤回。")
            col1, col2 = st.columns([1, 1])
            confirm_deleted = False
            with col1:
                if st.button("✔ 确认", use_container_width=True):
                    json_files = glob.glob(os.path.join(config_dir, "*.json"))

                    # 遍历删除
                    for file in json_files:
                        os.remove(file)

                    confirm_deleted = True
                    st.session_state["confirm_delete"] = False
            with col2:
                if st.button("❌ 取消", use_container_width=True):
                    pass


# 2 上传文件区 -----------------------------------------------------------------------------------------------------------


st.sidebar.subheader("📤 上传数据", )
st.sidebar.markdown('')

uploaded_files_dict = {}

uploaded_files = st.sidebar.file_uploader(
    "支持多文件上传, 默认最后一列为结果, 其余列为原因",
    type=["csv", "txt", "xlsx"],
    accept_multiple_files=True,
    help='目前 beta 版仅支持形状为两列的数据'
)

file_sep = st.sidebar.text_input(r"文件分隔符 (如果有)", placeholder="输入分隔符, 不需要加引号", help="常见的分隔符有: <;>, </t>, </s>, 输入不用带<>")

has_header = st.sidebar.checkbox("文件包含列名 (将自动去除)", value=False)

st.sidebar.markdown('---')

    # 2-1 尝试加载上传的文件 -----------------------------------------------------------------------------------------------------------------

if uploaded_files:
    with st.spinner('Loading drop files...'):
        for uploaded_file in uploaded_files:

            try:
                df = pd.read_csv(uploaded_file, sep=file_sep, header=0 if has_header else None)

            except ValueError:

                try:

                    df = pd.read_excel(uploaded_file, header=0 if has_header else None)

                except Exception as e:

                    st.warning(f"❌ 读取文件 `{uploaded_file.name}` 出错: {e}")
            else:

                if df.shape[1] < 2:
                    st.warning(f"⚠️ 文件 `{uploaded_file.name}` 小于两列，已跳过")
                    continue
                uploaded_files_dict[uploaded_file.name] = df

    use_uploaded_data = True

st.sidebar.markdown('\n')


# 3 预处理 ---------------------------------------------------------------------------------------------------------------


use_data_options = ["模拟样本", "本地样本", "上传样本"]
use_data_type = st.sidebar.selectbox("**选择分析的数据种类**", options=use_data_options, key="use_data_type")

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

    "to_DPLS_pred": {
    "cv": {
        "type": "slider",
        "label": "多重检验折数",
        "min": 1, "max": 10, "step": 1, "value": 5
    },
    "max_iter": {
        "type": "slider",
        "label": "DPLS最大迭代层",
        "min": 1, "max": 500, "value": 20
    },
    "R_mode": {
        "type": "selectslider",
        "label": "求R模式",
        "help": "[fusion]: 返回整个样本集的DPLSR, [single]: 返回每列的DPLSR",
        "options": ['fusion', 'single'],
        "value": 'fusion'
    },
    "distance_pattern": {
        "type": "selectbox",
        "label": "距离矩阵种类",
        "help": "[Euc]:欧氏距离. [Mah]:曼哈顿距离, [Pairs]:成对组合距离, [Ming]:闵氏距离",
        "options": ["Euc", 'Mah', 'Pairs', 'Ming'],
        "value": "Euc"
    },
    "whiten": {
        "type": "checkbox",
        "label": "标准化",
        "value": False
    },
    "square": {
        "type": "checkbox",
        "label": "距离矩阵左乘自己的转置",
        "value": False
    },
},

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

# 3-3 选择预处理
preprocess_selection = st.sidebar.multiselect(
    label="预处理流程（可多选）", key="preprocess_selection",
    options=list(process.keys())
)

# 3-4 预处理的打印内容
if preprocess_selection:
    preprocess_print = {p + 1: process_ for p, process_ in enumerate(preprocess_selection)}
else:
    preprocess_print = {}


# 4 方法 ----------------------------------------------------------------------------------------------------------------


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


def gen_DPLSR_controller(dict_key: int | str, expanded=True):
    DPLS_kwargs = param_controller(
        param_list=['DPLSR'],
        para_descriptions={'DPLSR': method_descriptions.get('DPLSR')},
        param_controls={"DPLSR": DPLSR_param_dict.copy()},
        desc='方法',
        a_copied_dict=dict_key, expanded=expanded
    )
    return DPLS_kwargs

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

# 4-3 选择方法
method_selection = st.sidebar.multiselect(
    label="分析方法（可多选）", key="create_method_selection",
    options=list(algorithms.keys())
)

# 4-4 方法的打印内容
if method_selection:
    method_print = {m + 1: method_ for m, method_ in enumerate(method_selection)}
else:
    method_print = {}


# 5 分类 ----------------------------------------------------------------------------------------------------------------


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

# 5-4 选择分类器
st.sidebar.markdown('---')
classify_selection = st.sidebar.multiselect("分类方法 (可多选)", ['All']+list(classify_dict.keys()), key="classify_selection")
if "All" in classify_selection:
    classify_selection = list(classify_dict.keys())

# 5-5 侧边栏分类器输入区
classify_shuffle_1, classify_shuffle_2 = st.sidebar.columns([1, 1])
with classify_shuffle_1:
    cv_mode = st.selectbox('训练集选取模式', options=['uniform', 'layers'], key="cv_mode")

with classify_shuffle_2:
    classify_shuffle_seed = st.number_input('样本混淆种子', value=42, key="classify_shuffle_seed")

st.sidebar.markdown('')
classify_cv_col, classify_cv_mode_col = st.sidebar.columns([2, 1])

with classify_cv_col:
    classify_cv = st.slider('多重检验折数', min_value=1, max_value=10, value=5, step=1, key="classify_cv")

with classify_cv_mode_col:
    st.markdown("<div style='height:37px;'></div>", unsafe_allow_html=True)
    classify_shuffle_confirm = st.checkbox('混淆样本顺序', value=False, key="classify_shuffle_confirm")

# 5-6 分类器打印内容
if classify_selection:
    classify_print = {r + 1: classify_ for r, classify_ in enumerate(classify_selection)}
else:
    classify_print = {}



# 6 线程与运行按钮 --------------------------------------------------------------------------------------------------------


st.sidebar.markdown('---')
st.sidebar.markdown('')
Thread = st.sidebar.slider("并行线程数: 轻量任务以及报错时设置值 = 1", 1, 16, 1, key='Thread')
st.sidebar.markdown('')
run_button = st.sidebar.button("**Analysis GO**", use_container_width=True, icon='🚀')


# 7 数据分析种类确定 -------------------------------------------------------------------------------------------------------

use_files_dict={}
with host_panel_setting:

    # render_dataset_title('样本')
    st.subheader("样本")

    st.markdown('---')

    # 7-0-1 定义公共功能区 ------------------------------------------------------------------------------------------------

    total_0 = 0
    total_1 = 0
    total_file = 0

    # 7-1 分析本地数据-----------------------------------------------------------------------------------------------------

    if use_data_type == use_data_options[1]:

        # 3-1-1 读取本地数据 ----------------------------------------------------------------------------------------------

        local_datasets = [name for name in os.listdir(local_data_dir) if os.path.isdir(os.path.join(local_data_dir, name))]

        local_datasets = sorted(local_datasets)

        # 读取本地数据的描述文件
        database_descriptions = {dataset_dir: read_txt_or_default(os.path.join(local_data_dir, dataset_dir)) for
                                 dataset_dir in
                                 local_datasets}

        # 7-1-2 定义分析本地数据的控制面板 -----------------------------------------------------------------------------------

        local_file_panel_expander = st.expander("**DPLS_lab**", expanded=True)

        with local_file_panel_expander:

            st.markdown('')
            st.markdown('')
            st.markdown('---')

            # 选择要分析的数据集
            database_selections = st.multiselect(label='分析本地数据集 (可多选)',
                                                 key="database_selections",
                                                 options=list(local_datasets),
                                                 default=list(local_datasets)[0])

            data_sort_mode = st.selectbox(label='文件排序',
                                                 key="data_sort_mode",
                                                 options=["按首字母", "按大小"],
                                                 )

            Database_print = '-'.join(database_selections)

            illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False])

            direction_panel = st.columns([1, 1])

            # 文件方向
            with direction_panel[0]:
                file_direction = st.selectbox("样本方向", ["AB&BA", "AB", "BA", ], key="file_direction")

            # 分析方向
            with direction_panel[1]:
                analys_direction = st.selectbox("分析方向", ["AB&BA", "AB", "BA"], key="analys_direction")

            col_file_num, col_test_seed = st.columns([1, 1])

            with col_file_num:
                test_files_num = st.number_input(
                    "文件抽样数",
                    min_value=10,
                    max_value=10000,
                    value=50,
                    step=10
                    , key="test_files_num")

            with col_test_seed:
                seed_value = st.number_input(
                    "文件抽样种子",
                    min_value=0,
                    max_value=20000,
                    value=42,
                    step=1
                    , key="seed_value")

            thresh_range = st.slider("样本数限制在", 0, 10000, (100, 1500), key="thresh_range", step=100)

            thresh_range_col, show_DPLS_pred_col = st.columns([2.2, 1])
            # 检视按钮
            with thresh_range_col:
                check_file_button = st.button("**检视**", use_container_width=True, icon="🔍")
            with show_DPLS_pred_col:
                st.markdown("<div style='height:1px'></div>", unsafe_allow_html=True)
                print_pred = st.checkbox('show DPLS', value=False, key='print_pred')

        # 7-1-3 在 detail 区打印控制面板的参数 ------------------------------------------------------------------------------

        with host_panel_detail:

            # 打印数据集与数据集描述
            st.markdown(f"""
                            <div style="
                                font-family: 'Segoe UI Variable Text', 'Roboto', 'Helvetica Neue', sans-serif !important;
                                font-size: 21px;
                                font-weight: 600;
                            ">
                                数据集:\t{database_selections}
                            </div>
                            """, unsafe_allow_html=True)

            st.markdown('')

            st.markdown(f"""
                    <div style="
                        padding-left: 1px;
                        font-family: 'Segoe UI Variable Text', 'Roboto', 'Helvetica Neue', sans-serif !important;
                        font-size: 17px;
                        font-weight: 270;
                    ">
                        {
    
            database_descriptions.get(database_selections[0], '') if len(database_selections) == 1 else "多个数据集"
    
            }
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('---')

        # 7-1-4 收集参数 -------------------------------------------------------------------------------------------------

        file_param = {
            "正在使用": f"{use_data_type}",
            "非法样本兼容": illegal_compatible_mode,
            '样本方向': file_direction,
            '分析方向': analys_direction,
            '文件抽样数': test_files_num,
            '随机种子': seed_value,
            '样本量': thresh_range,

        }

        db_select_kwargs = {'relation': file_direction, 'threshold': list(thresh_range), 'seed': seed_value,'test_SAMPLE': test_files_num}

        # 7-1-5 定义检视样本逻辑 -------------------------------------------------------------------------------------------

        if check_file_button or run_button:

            for database in database_selections:
                read_files, file_names, files_cause = data_presenter(database, **db_select_kwargs)
                use_files_dict[database] = {"files_pair": dict(zip(file_names, read_files)), "files_cause": dict(zip(file_names, files_cause))}
                count_0 = files_cause.count(0)
                count_1 = files_cause.count(1)

                total_0 += count_0
                total_1 += count_1
                total_file += len(read_files)

    if use_data_type == use_data_options[0]:

        Database_print="Created_Files"

        # 7-2-1 定义模拟数据参数控制面板 ------------------------------------------------------------------------------------

        with host_panel_detail:

            st.markdown(f"""
            <div style="
                font-family: 'Segoe UI Variable Text', 'Roboto', 'Helvetica Neue', sans-serif !important;
                font-size: 21px;
                font-weight: 600;
            ">
                已选择的数据集:\t模拟数据
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            st.markdown(f"""
            <div style="
                padding-left: 1px;
                font-family: 'Segoe UI Variable Text', 'Roboto', 'Helvetica Neue', sans-serif !important;
                font-size: 17px;
                ont-weight: 270;
            ">
                模拟数据
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <hr style="margin-top: 35px; border: none; border-top: 1px solid #d3d3d3;" />
            """, unsafe_allow_html=True)

        # 7-2-2 与 本地文件模式共用的功能部分 --------------------------------------------------------------------------------

        create_file_panel_expander = st.expander("**DPLS_lab**", expanded=True)

        with create_file_panel_expander:
            st.markdown('')
            st.markdown('')
            st.markdown("---")

            illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False], key="illegal_compatible_mode")

            direction_panel = st.columns([1, 1])

            # 文件方向
            with direction_panel[0]:
                file_direction = st.selectbox("样本方向", ["AB&BA", "AB", "BA", ], key="file_direction")

            # 分析方向
            with direction_panel[1]:
                analys_direction = st.selectbox("分析方向", ["AB&BA", "AB", "BA"], key="analys_direction")

            col_file_num, col_test_seed = st.columns([1, 1])

            with col_file_num:
                test_files_num = st.number_input(
                    "文件抽样数",
                    min_value=10,
                    max_value=10000,
                    value=50,
                    step=10
                    , key="test_files_num")

            with col_test_seed:
                seed_value = st.number_input(
                    "文件抽样种子",
                    min_value=0,
                    max_value=20000,
                    value=42,
                    step=1
                    , key="seed_value")

            thresh_range = st.slider("样本数限制在", 0, 10000, (100, 1500), key="thresh_range", step=100)
            st.markdown("")


# 7-3 定义[上传]数据参数控制面板 --------------------------------------------------------------------------------------------


    if use_data_type == use_data_options[2]:

        Database_print = "Uploaded_Files"

        with host_panel_detail:

            if  uploaded_files:
                st.markdown(f"""
                <div style="
                    font-family: 'Segoe UI Variable Text', 'Roboto', 'Helvetica Neue', sans-serif !important;
                    font-size: 21px;
                    font-weight: 600;
                ">
                    已选择的数据集:\t上传的数据 ({len(uploaded_files)})
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div style="
                    font-family: 'Segoe UI Variable Text', 'Roboto', 'Helvetica Neue', sans-serif !important;
                    font-size: 21px;
                    font-weight: 600;
                ">
                    已选择的数据集:\t 无上传的数据
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown(f"""
            <div style="
                padding-left: 1px;
                font-family: 'Segoe UI Variable Text', 'Roboto', 'Helvetica Neue', sans-serif !important;
                font-size: 17px;
                ont-weight: 270;
            ">
                上传的数据 ({len(uploaded_files)})
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

        files_cause = ['Unknown'] * len(use_files_dict.keys())
        use_files_dict["Upload_files"] = {"files_pair": uploaded_files_dict, "files_cause":dict(zip(use_files_dict.keys(), files_cause))}
        total_file = len(use_files_dict)

        # 7-3-2 定义[上传]数据参数控制面板 ----------------------------------------------------------------------------------

        upload_file_panel_expander = st.expander("**DPLS_lab**", expanded=True)

        with upload_file_panel_expander:
            st.markdown('')
            st.markdown('')
            st.markdown("---")

            illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False], key="illegal_compatible_mode")

            direction_panel = st.columns([1, 1])

            # 文件方向
            with direction_panel[0]:
                file_direction = st.selectbox("样本方向", ["AB&BA", "AB", "BA", ], key="file_direction")

            # 分析方向
            with direction_panel[1]:
                analys_direction = st.selectbox("分析方向", ["AB&BA", "AB", "BA"], key="analys_direction")

            col_file_num, col_test_seed = st.columns([1, 1])

            with col_file_num:
                test_files_num = st.number_input(
                    "文件抽样数",
                    min_value=10,
                    max_value=10000,
                    value=50,
                    step=10
                    , key="test_files_num")

            with col_test_seed:
                seed_value = st.number_input(
                    "文件抽样种子",
                    min_value=0,
                    max_value=20000,
                    value=42,
                    step=1
                    , key="seed_value")

            file_param = {
                "正在使用": f"{use_data_type}",
                "非法样本兼容": illegal_compatible_mode,
            }

            thresh_range_col, show_DPLS_pred_col = st.columns([2.2, 1])
            # 检视按钮
            with thresh_range_col:
                check_file_button = st.button("**检视**", use_container_width=True, icon="🔍")
            with show_DPLS_pred_col:
                st.markdown("<div style='height:1px'></div>", unsafe_allow_html=True)
                print_pred = st.checkbox('show DPLS', value=False, key='print_pred')


# 模拟样本的独立功能区, 单独于 host_panel_setting 外 -------------------------------------------------------------------------


if use_data_type == use_data_options[0]:

    st.markdown('')
    st.markdown('')
    st.markdown('')
    hr_second(dark_color="#244690", height=2, light_color="#26519d")
    st.markdown('')
    st.markdown('')
    st.markdown('')

    # 7-2-3 定义生成模拟数据的控制面板 ---------------------------------------------------------------------------------------

    create_preview_panel, create_sep1, create_control_panel = st.columns([.75, 0.013, .25])

    st.markdown('')
    st.markdown('')
    st.markdown('---')
    st.markdown('')
    st.markdown('')

    create_check_file_expander = st.expander("检视预览函数样本", expanded=False)

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

    with create_preview_panel:

        st.subheader('模拟样本预览区')
        st.markdown('')
        st.markdown('---')
        st.markdown("")

        # 7-5-1 定义内容区 -----------------------------------------------------------------------------------------------

        formular_refresh_col, formular_sep_1, formular_title_add_x, add_x_col, minus_x_col,  formular_sep_2, formular_title_add_xtox, add_xtox_col, minus_xtox_col, formular_sep_3  = st.columns([.6, .3, .75, .21, .21, .3, .75, .21, .21, 3.5])

        # with formular_title_col:
        #     render_section_title("你的函数 be like :", underline=False)

        with formular_refresh_col:
            f_refresh_button = st.button("Clear", use_container_width=True)
            if f_refresh_button:

                st.session_state["formular_refresh"]= True

                st.rerun()

        with formular_sep_1:

            st.markdown("""
                <div style="
                    width: 1px;
                    height: 50px;
                    background-color: #cccccc;
                    margin: auto;
                    margin-top: -5px;
                "></div>
            """, unsafe_allow_html=True)

        with formular_title_add_x:

            render_noline_title("快速添加一个独立项")

        st.session_state['add_x'] = False
        st.session_state['add_xtox'] = False
        st.session_state['minus_x'] = False
        st.session_state['minus_xtox'] = False

        with add_x_col:
            add_x = st.button("＋", key="add_x_b")
            if add_x:
                st.session_state["add_x"] = True

        with minus_x_col:
            minus_x = st.button("－", key="minus_x_b")
            if minus_x:
                st.session_state["minus_x"] = True

        with formular_sep_2:

            st.markdown("""
                <div style="
                    width: 1px;
                    height: 50px;
                    background-color: #cccccc;
                    margin: auto;
                    margin-top: -5px;
                "></div>
            """, unsafe_allow_html=True)

        with formular_title_add_xtox:
            render_noline_title("快速添加一个互作项")
        with add_xtox_col:
            add_xtox = st.button("＋", key="add_xtox_b")
            if add_xtox:
                st.session_state["add_xtox"] = True
        with minus_xtox_col:
            minus_xtox = st.button("－", key="minus_xtox_b")
            if minus_xtox:
                st.session_state["minus_xtox"] = True



    with create_control_panel:

        st.subheader('函数控制面板')
        st.markdown('')
        st.markdown('---')
        st.markdown("")

        create_data_expander = st.expander('**生成模拟数据**', expanded=True)

        create_help_dict = {

            "create_param_num":'这个参数决定 [特征池] 的大小,  参与 [y] 构成的那些自变量 [X] 将会在池中选择, 池中剩余的 [特征] 为[无关特征]',
            "create_use_x_num": "从 [特征池] 内取用的 [X] 的数量, 取用 [X] 数量 ≤ [特征池]大小",
            "create_x_num": "此参数定义了构成 [y] 的函数关系内有多少独立项, 如 [y = x_1 + x_1^2 + x_2 + (x_1×x_2)],  \n"
                                                        "此时 [独立项] = 3 (即除 (x_1×x_2) 项以外的项, 他们仅由一个 [x] 决定), 注意到这里出现了重复的 [x]: x_1,"
                                                        "因为此参数不关注 [x] 是否重复,  \n但受参数: [使用的X数量] 的约束, "
                                                        "因为被标记为需要使用的 [X] 必须在 [y] 的公式中至少出现一次, ",

            "create_redun":"如果此项为 [True], 则可以有 [x非独立生成], 以下提到的情况就是 [合理] 的:  \n"
                                                     " 设参数 [独立项数量] = 3 , 生成的函数关系为 [y=x_1 + 2×x_2 + 3×x_3]"
                                                     " 表面上 [3×x_3] 这一项仅由 [x_3]构成, 因此被视为 [独立项], "
                                                     "但 [x_3] 由关系: [x_3=x_1×x_2] 约束, 所以 [真实] 的函数关系为  \n"
                                                     " [y=x_1 + 2×x_2 + 3×(x_1×x_2))],实际上只有两个 [X]参与了函数关系  \n"
                                                     "[⚠️激进的参数]: 此项将会显著增加函数复杂度  \n"
                                                     "[⚡不受控的参数]: 开启此项后, 将不受随机种子控制",

            "create_xtox_num": "此参数定义了构成 [y] 的函数关系内有多少互作项, 互作列是指由[多个X]影响的列,"
                                                                   " 只会由选择为[使用的X]构成,  \n如 [y = x_1 + x_1^2 + x_2 + (x_1×x_2)], "
                                                                "此时 [互作项] = 1",

            "create_xtox_level":"此项只限制了每个互作项的 [最高参与项数], 而 [不一定] 得到最高参与项,  \n"
                                                             "如设置此项数为 4, 得到 [x_3=x_1×x_2×(x_2-x_1)], 此时 [x_3] 的项数 [=4],  \n"
                                                             "但是更换随机种子得到 [x_3=x_1×x_2], 此时 [2<=4], 是合理的.  \n"
                                                             "[总结]: 当此项设为 [n] 时, 将不会超过参与项超过 [n] 的 [互作项]",

            "create_funcseed": "输入x数量后, 每个x被分到一个f构成f(x), f不一定一样, 随机种子用于决定这些 f",

            "create_x_bank":                                            "'正弦函数': f'sin(πx)',  \n"
                                           "'余弦函数': f'cos(πx)'  \n"
                                           "'二次函数': f'2(x)^2',  \n"
                                           "'平方根函数': f'sqrt(x)',  \n"
                                           "'指数函数': f'e^x',  \n"
                                           "'对数函数（平移）': f'log(x+1)',  \n"
                                           "'对数函数（加偏移防负值）': f'log(x)',  \n"
                                           "'Sigmoid 函数': f'1/(1+e^-6x)',  \n"
                                           "'三次多项式函数': f'2x^3+x^2-2x',  \n"
                                           "'指数幂函数': f'2^5(x+1)',  \n"
                                           "'高频正弦函数': f'sin(2πx)',  \n"
                                           "'混合三角+线性函数': f'1/5×sin(4x)+(11/10)×x',  \n"
                                           "'高频正弦 + 线性项': f'sin(5πx)+x',  \n"
                                           "'高频余弦函数': f'cos(6πx)',  \n"
                                           "'高频正弦线性混合函数': f'1/10×sin(10.6×f)+(11/10)×x',  \n"
                                           "'非线性频率余弦函数': f'cos(5πx(x+1))',  \n"
                                           "'非线性频率正弦函数': f'sin(4πx(x+1))'  \n",

        }

        with create_data_expander:

            # 7-2-3-1 无关功能 -------------------------------------------------------------------------------------------

            st.markdown('---')

            unrelated_expander = st.expander("**无关**", expanded=True)

            with unrelated_expander:

                create_use_x_col, create_x_col = st.columns([1, 1])

                with create_x_col:

                    create_interact_num = st.number_input("无关的X数量", min_value=0, max_value=20, step=1, value=0,
                                                       key="create_param_num",
                                                       help=create_help_dict.get("create_param_num","无描述")
                                                       )

                with create_use_x_col:

                    if "create_use_x_num" in st.session_state:

                        if st.session_state.get("add_x", False) or st.session_state.get("add_xtox", False):

                            create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                               max_value=20,
                                                               step=1, value=st.session_state["create_use_x_num"]+1, key="create_use_x_num",
                                                               help=create_help_dict.get("create_use_x_num", "无描述"),
                                                               )

                        elif st.session_state.get("minus_x", False) or st.session_state.get("minus_xtox", False):

                            if st.session_state["create_use_x_num"] > 1:
                                create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                                   max_value=20,
                                                                   step=1, value=st.session_state["create_use_x_num"]-1, key="create_use_x_num",
                                                                   help=create_help_dict.get("create_use_x_num", "无描述"),
                                                                   )
                            else:

                                create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                                   max_value=20,
                                                                   step=1, value=1, key="create_use_x_num",
                                                                   help=create_help_dict.get("create_use_x_num", "无描述"),
                                                                   )

                        else:

                            create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                               max_value=20,
                                                               step=1, value=st.session_state["create_use_x_num"], key="create_use_x_num",
                                                               help=create_help_dict.get("create_use_x_num", "无描述"),
                                                               )



                    else:

                        create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                           max_value=20,
                                                           step=1, value=2, key="create_use_x_num",
                                                           help=create_help_dict.get("create_use_x_num", "无描述"),
                                                           )




                    create_param_num = create_interact_num + create_use_x_num

            # 7-2-3-2 冗余功能 -------------------------------------------------------------------------------------------

            redun_expander = st.expander("**冗余**", expanded=True)

            with redun_expander:

                create_x_num_col, create_redun_col = st.columns([1, 1])

                with create_x_num_col:


                    if st.session_state.get("create_xtox_num", 0) == 0:

                        create_x_limit = 1

                    else:
                        create_x_limit = 0

                    if "create_x_num" in st.session_state:

                        if st.session_state.get("add_x", False):

                            create_x_num = st.number_input("独立项数量", min_value=create_x_limit, max_value=10, step=1,
                                                              value=st.session_state["create_x_num"] + 1,
                                                              key="create_x_num",
                                                              help=create_help_dict.get("create_x_num", "无描述"),
                                                              )

                        elif st.session_state.get("minus_x", False) and st.session_state["create_x_num"] > create_x_limit:

                            create_x_num = st.number_input("独立项数量", min_value=create_x_limit, max_value=10, step=1,
                                                              value=st.session_state["create_x_num"] - 1,
                                                              key="create_x_num",
                                                              help=create_help_dict.get("create_x_num", "无描述"),
                                                              )

                        else:

                            create_x_num = st.number_input("独立项数量", min_value=create_x_limit, max_value=10, step=1,
                                                              value=st.session_state["create_x_num"],
                                                              key="create_x_num",
                                                              help=create_help_dict.get("create_x_num", "无描述"),
                                                              )


                    else:

                        create_x_num = st.number_input("独立项数量", min_value=0, max_value=20, step=1,
                                                       value=2,
                                                       key="create_x_num",
                                                       help=create_help_dict["create_x_num"],
                                                       )


                    create_redun_count = st.slider("X的冗余最大层 ⚠️", min_value=1, max_value=8, step=1, value=5,
                                                   key="create_redun_count",
                                                   help="[⛓️‍存在依赖项]: 此项发挥作用需要 [开启冗余] 被设置为 [True],  \n"
                                                        "[⚠️激进的参数]: 此参数将显著影响函数复杂度,  \n"
                                                        "[互作层级]: 指原变量外嵌套的 f 层级, 如 x_3 = sin(x_2), 此时 [嵌套层级] = 1  \n"
                                                        " x_3 = cos(sin(x_2))^2, 此时 [嵌套层级] = 3  \n"
                                                        "[互作层级] 越高,则越可能生成复杂的 [冗余项]  \n"
                                                   )

                with create_redun_col:

                    create_redun = st.selectbox("**开启冗余** ⚠️", [False, True], key="create_redun",
                                                help=create_help_dict.get("create_redun", "无描述"),)

                    create_redun_slope = st.slider("X的冗余倾向 ⚠️", min_value=1.0, max_value=8.0, step=0.5, value=2.5,
                                                   key="create_redun_slope",
                                                   help="[⛓️‍存在依赖项]: 此项发挥作用需要 [开启冗余] 被设置为 [True],  \n"
                                                        "[⚠️激进的参数]: 此参数将显著影响函数复杂度,  \n"
                                                        "[冗余倾向] 越高,则越可能生成复杂的 [冗余项], 也越可能使用 [冗余特征] 参与构成 [y]  \n"
                                                   )

            # 7-2-3-3互作功能 --------------------------------------------------------------------------------------------

            interaction_expander = st.expander("**互作**", expanded=True)

            with interaction_expander:

                create_xtox_num_col, create_xtox_level_col = st.columns([1, 1])


                with create_xtox_num_col:

                    if st.session_state.get("create_x_num", 2) == 0:

                        create_xtox_limit = 1

                    else:
                        create_xtox_limit = 0


                    if "create_xtox_num" in st.session_state:


                        if st.session_state.get("add_xtox", False):

                            create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10, step=1,
                                                              value=st.session_state["create_xtox_num"] + 1,
                                                              key="create_xtox_num",
                                                              help=create_help_dict.get("create_xtox_num", "无描述"),
                                                              )

                        elif st.session_state.get("minus_xtox", False) and st.session_state["create_xtox_num"] > create_xtox_limit:

                            create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10, step=1,
                                                              value=st.session_state["create_xtox_num"] - 1,
                                                              key="create_xtox_num",
                                                              help=create_help_dict.get("create_xtox_num", "无描述"),
                                                              )

                        else:

                            create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10, step=1,

                                                              value=st.session_state["create_xtox_num"],
                                                              key="create_xtox_num",
                                                              help=create_help_dict.get("create_xtox_num", "无描述"),
                                                              )

                    else:
                        create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10, step=0, value=0,
                                                          key="create_xtox_num",
                                                          help=create_help_dict.get("create_xtox_num", "无描述"),
                                                          )

                with create_xtox_level_col:

                    create_xtox_level = st.number_input("互作最高参与项", min_value=2, max_value=5, step=1, value=2,
                                                        key="create_xtox_level",
                                                        help=create_help_dict.get("create_xtox_level","无描述"),
                                                        )

            # 7-2-3-4 随机种子 -------------------------------------------------------------------------------------------

            creation_seed_expander = st.expander("**随机种子**", expanded=True)

            with creation_seed_expander:

                create_funcseed_col, create_xseed_col = st.columns([1, 1])


                with create_funcseed_col:

                    if "formular_refresh" in st.session_state and "create_funcseed" in st.session_state:

                        if st.session_state["formular_refresh"]:

                            create_funcseed = st.number_input("f(x)随机种子", min_value=0, max_value=20000, step=1, value=st.session_state["create_funcseed"]+1, key="create_funcseed",
                                                           help=create_help_dict.get("create_funcseed", "无描述"),
                                                           )

                        else:

                            create_funcseed = st.number_input("f(x)随机种子", min_value=0, max_value=20000, step=1,
                                                              value=st.session_state["create_funcseed"],
                                                              key="create_funcseed",
                                                              help=create_help_dict.get("create_funcseed", "无描述"),
                                                              )

                        st.session_state["formular_refresh"] = False


                    else:

                        create_funcseed = st.number_input("f(x)随机种子", min_value=0, max_value=20000, step=1, value=73, key="create_funcseed",
                                                       help=create_help_dict.get("create_funcseed", "无描述"),
                                                       )
                        st.session_state["formular_refresh"] = False


                    create_redun_seed = st.number_input("冗余随机种子", min_value=0, max_value=20000, step=1, value=73, key="create_redun_seed",
                                                       help="开启冗余后, 随机种子用于决定冗余的状态"
                                                       )



                with create_xseed_col:

                    create_xseed = st.number_input("x的随机种子", min_value=0, max_value=20000, step=1, value=409, key="create_xseed",
                                                   help="输入x数量后, 每个x会在你给出的定义域上随机抽取n个(每个x抽到的不一样, 即使你只给了一个随机种子), x的随机种子用于决定这个抽取过程"
                                                   )

                    create_x_mode = st.selectbox("X取值模式", ["均匀", "成长型", "抛物线型"], key="create_x_mode",
                                                help="[均匀]模式时,X在定义域内取到每个点的概率相同, [成长型]:X越大取到的概率越大, [抛物线型]:中间的X被取到的概率更大")

                    create_x_mode_transfer = {
                        "均匀":"uniform",
                        "成长型":"grow",
                        "抛物线型":'parabola'
                    }

                    create_x_mode = create_x_mode_transfer[create_x_mode]


            # 7-2-3-5 一些其他功能 ----------------------------------------------------------------------------------------

            create_noise = st.slider("y_obs 噪音强度", min_value=0.0, max_value=8.0, step=0.1, value=0.5,
                                     key="create_noise",
                                     help="生成模拟样本后往[期望值y_exp]添加的噪音强度"
                                     )

            st.markdown("")
            create_linear_coef_col, create_linear_intercept_col = st.columns([1, 1])

            with create_linear_coef_col:



                create_linear_coef_range = st.slider("线性函数系数范围", -10, 10, (-3, 3), key="create_linear_coef_range", step=1)


            with create_linear_intercept_col:

                create_linear_intercept_range = st.slider("线性函数截距范围", -25, 25, (-5, 5), key='create_linear_intercept_range', step=2)

            create_x_bank = st.multiselect(label="可出现的f(x):", options=["All"] + list(function_dict.keys()),
                                           default=["线性函数", "正弦函数", "二次函数"], key="create_x_bank",
                                           help= create_help_dict.get("create_x_bank", "无描述")

                                           )

            if "All" in create_x_bank:
                create_x_bank = list(function_dict.keys())

            if not create_x_bank:
                create_x_bank = ["正弦函数"]


            create_xtox_bank = st.multiselect(label="可出现的互作函数:", options=["All"] + list(xtox_func_dict.keys()),
                                              default=["积函数", "绝对值和函数", "正弦和函数"], key="create_xtox_bank")

            if "All" in create_xtox_bank:
                create_xtox_bank = list(xtox_func_dict.keys())


            create_define = st.slider("定义域", -5.0, 5.0, (-1.0, 1.0), key="create_define", step=0.1)

            create_define_left = create_define[0]
            create_define_right = create_define[1]

            create_thresh_range = st.slider("数据量限制在", 0, 10000, (300, 700), key="create_thresh_range", step=100)

            create_linear_limit_col, only_usedx_col = st.columns([2, 1])

            with create_linear_limit_col:

                create_linear_limit = st.checkbox("仅生成线性样本", key="create_linear_limit")

            with only_usedx_col:

                use_x_piked = st.checkbox("排除无关特征", key="use_x_piked", value=True)

            thresh_range_col, show_DPLS_pred_col = st.columns([2.2, 1])
            # 检视按钮
            with thresh_range_col:
                check_file_button = st.button("**检视**", use_container_width=True, icon="🔍")
            with show_DPLS_pred_col:
                st.markdown("<div style='height:1px'></div>", unsafe_allow_html=True)
                print_pred = st.checkbox('show DPLS', value=False, key='print_pred')


        # 7-3 收集参数 ---------------------------------------------------------------------------------------------------

        # 传给 gen_y_exp 的参数
        create_kwargs = {

            "param_num": create_param_num,
            "use_x_num": create_use_x_num,
            "x_num": create_x_num,
            "x_to_x_num": create_xtox_num,
            "x_to_x_level": create_xtox_level,
            "redundancy": create_redun,
            "redun_ratio": create_redun_slope,
            "func_seed": create_funcseed,
            "x_seed": create_xseed,
            "linear_coef_range": create_linear_coef_range,
            "linear_intercept_range": create_linear_intercept_range,
            "x_start": create_define_left,
            "x_end": create_define_right,
            "linear": create_linear_limit,
            "use_x_func": create_x_bank,
            "use_xtox_func": create_xtox_bank,
            'redun_seed': create_redun_seed,
            'x_mode': create_x_mode,
            'redun_max_count': create_redun_count,
        }

        # 传给 detail_panel 的参数
        file_param = {
            "正在分析": "模拟样本",
            "非法样本兼容": illegal_compatible_mode,
            '样本方向': file_direction,
            '分析方向': analys_direction,
            '文件抽样数': test_files_num,
            '文件抽样种子': seed_value,
            '样本量': thresh_range,
            "开启冗余": f"是" if create_redun else "否",
            "仅生成线性样本": create_linear_limit,
        }

    # 7-4 定义模拟数据检视逻辑 ---------------------------------------------------------------------------------------------

    created_data = {}

    if check_file_button or run_button:

        with st.spinner('正在生模拟数据...'):

            create_times = test_files_num // (create_use_x_num + create_xtox_num)

            create_func_seeds = gen_seed(create_times, rand_seed=create_funcseed, gen_times=1)[0]
            create_x_seeds = gen_seed(create_times, rand_seed=create_xseed, gen_times=1)[0]

            for create_i in range(create_times):

                np.random.seed(create_func_seeds[create_i])

                create_kwargs["func_seed"] = create_func_seeds[create_i]
                create_kwargs["x_seed"] = create_x_seeds[create_i]

                eg_create_samples = np.random.randint(thresh_range[0], thresh_range[1])

                # 7-4-1 参数选择完毕, 开始生成模拟数据 ----------------------------------------------------------------------

                time.sleep(1)

                x_create, X_create, y_exp_create, x_picked_create = gen_y_exp(sample_num=eg_create_samples,
                                                                              **create_kwargs)
                np.random.seed(create_x_seeds[create_i])
                y_create_obs = y_exp_create + np.random.normal(size=y_exp_create.shape[0], loc=0, scale=y_exp_create.std() * create_noise)

                create_func_name = 'y=' + '+'.join(list(X_create))

                if use_x_piked:

                    create_i_use = x_create.copy()[x_picked_create]

                else:
                    create_i_use = x_create.copy()

                create_i_use["y"] = y_create_obs
                created_data_pair, created_pair_name, created_data_cause = return_cause_pair(create_i_use, relation=file_direction, prefix=f'[{create_func_name}]')

                created_data[create_func_name] = {"files_pair": dict(zip(created_pair_name, created_data_pair)),
                                           "files_cause": dict(zip(created_pair_name, created_data_cause))}

                count_0 = created_data_cause.count(0)
                count_1 = created_data_cause.count(1)

                total_0 += count_0
                total_1 += count_1
                total_file += len(created_data_pair)

        # 7-4-2 模拟数据存入全局变量: use_files_dict -----------------------------------------------------------------------

        use_files_dict = created_data

# 7-5 Createdata-enhance 区 ---------------------------------------------------------------------------------------------


    with create_preview_panel:

        with st.spinner("正在生成函数实例, 勿点击任何按钮..."):

            # 7-5-2 控制面板的参数传入用于 eg 的生成 -------------------------------------------------------------------------


            np.random.seed(create_kwargs["x_seed"])
            eg_create_samples = np.random.randint(create_thresh_range[0], create_thresh_range[1])

            if "eg_created" not in st.session_state:

                create_eg_kwargs = copy.deepcopy(create_kwargs)
                st.session_state["create_eg_kwargs"] = create_eg_kwargs
                eg_change=False

                x_eg, X_eg, y_exp_eg, x_picked_eg = gen_y_exp(**create_eg_kwargs, sample_num=eg_create_samples)

                st.session_state['x_eg'] = x_eg
                st.session_state["X_eg"] = X_eg
                st.session_state['y_exp_eg'] = y_exp_eg
                st.session_state['x_picked_eg'] = x_picked_eg
                st.session_state['eg_created'] = True

            else:

                eg_change = (st.session_state["create_eg_kwargs"] != create_kwargs)
                create_eg_kwargs = copy.deepcopy(create_kwargs)
                st.session_state["create_eg_kwargs"] = create_eg_kwargs

                if eg_change:

                    x_eg, X_eg, y_exp_eg, x_picked_eg = gen_y_exp(**create_eg_kwargs, sample_num=eg_create_samples)

                    st.session_state['x_eg'] = x_eg
                    st.session_state["X_eg"] = X_eg
                    st.session_state['y_exp_eg'] = y_exp_eg
                    st.session_state['x_picked_eg'] = x_picked_eg
                    st.session_state['eg_created'] = True

                else:

                    x_eg = st.session_state['x_eg']
                    X_eg = st.session_state['X_eg']
                    y_exp_eg = st.session_state['y_exp_eg']
                    x_picked_eg = st.session_state['x_picked_eg']


            # 7-5-3 生成 eg ---------------------------------------------------------------------------------------------

            if use_x_piked:

                x_eg_use = copy.deepcopy(x_eg)[x_picked_eg] # 坑爹的 df.copy

            else:
                x_eg_use = copy.deepcopy(x_eg)

            np.random.seed(create_kwargs["x_seed"])

            y_obs_eg = y_exp_eg + np.random.normal(size=y_exp_eg.shape[0], loc=0, scale=y_exp_eg.std() * create_noise)
            y_df_eg = pd.DataFrame()
            y_df_eg["y_exp"] = y_exp_eg
            y_df_eg["y_obs"] = y_obs_eg

            x_eg_columns = x_eg.columns.tolist()

        formula_eg = 'y=' + '+'.join(list(X_eg.columns))

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown('')
        st.markdown("")
        st.markdown("")
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        # 7-5-4 打印公式的latex格式 ---------------------------------------------------------------------------------------

        latex_expr = apply_colored_brackets(formula_eg)
        st.latex(latex_expr)

        st.markdown("")
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown("---")
        st.markdown('')
        st.markdown("")
        st.markdown("")
        st.markdown("")

        create_Formula_panel_1, create_Formula_sep_1, create_Formula_panel_2 = st.columns([.65,0.018, .35])

        # 独立与互作特征鉴别器(待升级)
        def count_unique_ai(expr: str, list_a: list) -> int:
            count = 0
            for item in list_a:
                if item in expr:
                    count += 1
            return count

        # 读取颜色（默认浅色）

        with create_Formula_panel_1:

            formula_panel1_col1, f_sep1, formula_panel1_col2 = st.columns([.5, 0.025, .5])

            with formula_panel1_col2:

                render_section_title('DPLS 参数')
                # 使用 expander 创建一个可折叠/展开的区域
                DPLSR_param_dict_ = DPLSR_param_dict.copy()
                create_DPLS_kwargs = param_controller(
                    param_list=['DPLSR'],
                    para_descriptions={'DPLSR': method_descriptions.get('DPLSR')},
                    param_controls={"DPLSR": DPLSR_param_dict_},
                    desc='方法',
                    a_copied_dict=True
                )

                if not create_DPLS_kwargs['DPLSR']["distance_pattern"]:
                    create_DPLS_kwargs['DPLSR']["distance_pattern"] = ['Euc']

            with formula_panel1_col1:

                render_section_title("参数与检测:")

                create_exe_expander_2 = st.expander(f"**{formula_eg[:200]}**", expanded=True)

                # 7-5-6-1 打印模拟样本的生成参数  -------------------------------------------------------------------------------

                with create_exe_expander_2:

                    st.markdown("---")
                    # principle_col, f_sep2, execute_col = st.columns([1, 0.1, 1])
                    #
                    # with principle_col:
                    #
                    #     unrelated_num = 0 if use_x_piked else create_param_num - create_use_x_num
                    #
                    #     col_level = [count_unique_ai(x_name, x_eg_use.columns) for x_name in X_eg.columns]
                    #
                    #     indi_col = col_level.count(1)
                    #     max_level = max(col_level)
                    #
                    #     render_dataset_title("自变量", 16)
                    #
                    #     principle_X = {
                    #
                    #         f"特征池 ": f"{create_param_num}",
                    #         f"使用的X ": create_use_x_num,
                    #         f"X的样本数": x_eg.shape[0],
                    #     }
                    #
                    #     display_detial_dict(principle_X)
                    #
                    #     render_dataset_title("无关", 16)
                    #
                    #     principle_unrelated = {
                    #
                    #         f"无关的X数量": unrelated_num,
                    #
                    #     }
                    #
                    #     display_detial_dict(principle_unrelated)
                    #
                    #     render_dataset_title("互作", 16)
                    #
                    #     principle_react = {
                    #
                    #         "独立项数量": create_x_num,
                    #         "互作项数量": create_xtox_num,
                    #         "互作项最高参与项数": create_xtox_level,
                    #     }
                    #
                    #     display_detial_dict(principle_react)
                    #
                    #     principle_redun = {
                    #
                    #         "是否开启冗余": "".join(["是" if create_redun else "否"]),
                    #     }
                    #
                    #     render_dataset_title("冗余", 16)
                    #     display_detial_dict(principle_redun)

                    # 7-5-6-1 打印参数的检测结果  -------------------------------------------------------------------

                    col_level = [count_unique_ai(x_name, x_eg_use.columns) for x_name in X_eg.columns]
                    indi_col = col_level.count(1)
                    iner_col = len(col_level) - indi_col
                    max_level = max(col_level)

                    unrelated_num = create_param_num - (indi_col+iner_col)

                    if unrelated_num > 0:
                        with formular_sep_3:
                            st.markdown(f"""
                            <div style='text-align: right; margin-top: -10px; padding-right: 20px;'>
                                <span style='color: #3580f5; font-size:25px;'><strong>[{unrelated_num}]</strong></span>
                                <span style='font-size:17px;'> 个未使用的自变量</span>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("")

                    x_use_xidx = [f'X_{x_eg_columns.index(x_i) + 1}' for x_i in x_picked_eg]
                    x_use_idx = [x_eg_columns.index(x_i) + 1 for x_i in x_picked_eg]

                    exe_X = {

                        f"X池: {', '.join([f'x_{i + 1}' for i in range(create_param_num)])}": " " + "".join(
                            ["✔" if create_param_num == create_param_num else "❌"]),
                        f"{'使用X: ' + ', '.join(x_use_xidx)}": "  " + "".join(
                            ["✔" if create_use_x_num == len(x_picked_eg) else "❌"]),
                        f"样本数 {create_thresh_range}: {x_eg.shape[0]}": " " + "".join(
                            ["✔" if (create_thresh_range[1] > x_eg.shape[0] > create_thresh_range[0]) else "❌"]),

                    }

                    display_detial_dict(exe_X)

                    render_dataset_title("无关", 15)
                    exe_unrelated = {

                        f"检测到无关列={(x_eg_use.shape[1] - create_use_x_num)}": " " + "".join(
                            ["✔" if unrelated_num == (x_eg_use.shape[1] - create_use_x_num) else "❌"]), }

                    display_detial_dict(exe_unrelated)

                    render_dataset_title("互作", 15)

                    exe_react = {

                        f"检测到独立项={indi_col}": " " + "".join(["✔" if (
                                indi_col == create_x_num) else "❌"]),
                        f"检测到互作项={X_eg.shape[1] - indi_col}": " " + "".join(["✔" if (
                                X_eg.shape[1] - indi_col == create_xtox_num) else "❌"]),
                        f"检测到的最高level={max_level}": " " + "".join(["✔" if (
                                max_level <= create_xtox_level) else "❌"]),

                    }

                    display_detial_dict(exe_react)

                    render_dataset_title("冗余", 15)

                    redun_detected = any(x_use_idx_i > create_param_num for x_use_idx_i in x_use_idx)

                    exe_redun = {

                        "检测到冗余": "".join(["是" if redun_detected else "未检出"]),

                    }

                    display_detial_dict(exe_redun)

                    if redun_detected:
                        display_detial_dict({f'X_{x_eg_columns.index(x_i) + 1}': x_i[:50] for x_i in x_picked_eg if
                                             x_eg_columns.index(x_i) > create_param_num - 1})

                    st.markdown("")


    with create_check_file_expander:

        formula_x_col, formula_check_file_sep, formula_y_col = st.columns([.738, 0.025, .382])

        with formula_x_col:

            st.markdown("")
            st.markdown("")
            render_section_title("来自函数的自变量:")

            with st.expander(f"**{x_eg.shape}**", expanded=True):

                st.markdown("---")
                st.dataframe(x_eg_use.copy(), height=795, hide_index=True)

        with formula_y_col:
            st.markdown("")
            st.markdown("")

            render_section_title("因变量")
            with st.expander(f"**{y_df_eg.shape}** | PersonR^2 = {calculate_corr(y_exp_eg, y_obs_eg)[0]:.3f}",
                             expanded=True):

                st.markdown("---")
                st.dataframe(y_df_eg, height=795, hide_index=True)


        # 7-5-5 打印样本的 X 与 y -----------------------------------------------------------------------------------------





        # 7-5-6 打印模拟样本的生成参数 与 参数符合度的检测结果 ------------------------------------------------------------------

        with create_Formula_panel_2:

            # 使用 expander 创建一个可折叠/展开的区域
            render_section_title("DPLS Plots:")


            with st.expander(f"**Plots** | PersonR^2: {calculate_corr(y_exp_eg, y_obs_eg)[0]:.3f}", expanded=True):

                st.markdown("---")

                enforce_P = st.slider("硬定位的 P: ", min_value=-1, max_value=create_DPLS_kwargs['DPLSR']['max_iter'] - 1, value=-1, key='enforce_P')

                plot_y_sep, plot_exp_col, plot_obs_col = st.columns([.58, 1.5, 1])

                with plot_exp_col:

                    print_exp = st.checkbox('y_exp', value=True, key='print_exp')

                with plot_obs_col:

                    print_obs = st.checkbox('y_obs', value=False, key='print_obs')

                st.markdown("")


            with st.expander("", expanded=True):

                with st.spinner("正在拟合..."):

                    pred_obj = DPLS(**create_DPLS_kwargs["DPLSR"]).fit(x_eg_use.copy(), y_df_eg['y_obs'].copy(),
                                                                       **create_DPLS_kwargs["DPLSR"])

                for x_i in x_picked_eg:

                    x = list(x_eg_use.columns).index(x_i)

                    if enforce_P == -1:

                        if create_DPLS_kwargs["DPLSR"]["R_mode"] == 'single':

                            y_pred = pred_obj.y_pred[x]
                            print("y_pred:", y_pred.shape)

                            R = pred_obj.R2[x]
                            P = pred_obj.p[x]

                        else:

                            y_pred = pred_obj.y_pred[0]

                            R = pred_obj.R2[0]
                            P = pred_obj.p[0]

                    else:

                        if create_DPLS_kwargs["DPLSR"]["R_mode"] == 'single':

                            y_pred = pred_obj.y_preds[x][:, enforce_P]
                            R = pred_obj.y_pred_R2[x][enforce_P]

                        else:

                            y_pred = pred_obj.y_preds[0][:, enforce_P]
                            R = pred_obj.y_pred_R2[0][enforce_P]

                        P = enforce_P

                    x_eg_use_i = pd.DataFrame()

                    x_eg_use_i[x_i] = pred_obj.X.copy()[:, x]

                    if print_exp:

                        x_eg_use_i['y_exp'] = y_df_eg['y_exp']

                    if print_obs:

                        x_eg_use_i['y_obs'] = y_df_eg['y_obs']

                    x_eg_use_i['preds'] = y_pred  # 使用完整预测值

                    # 创建一个空图
                    fig = go.Figure()

                    # 添加 y_exp（蓝色），优先添加以保证 preds 最上层
                    if print_exp:
                        fig.add_scatter(
                            x=x_eg_use_i[x_i],
                            y=x_eg_use_i['y_exp'],
                            mode='markers',
                            name='y_exp',
                            marker=dict(
                                color='#3580F5',
                                size=6,
                                opacity=0.75
                            )
                        )

                    # 添加 y_obs（绿色）
                    if print_obs:
                        fig.add_scatter(
                            x=x_eg_use_i[x_i],
                            y=x_eg_use_i['y_obs'],
                            mode='markers',
                            name='y_obs',
                            marker=dict(
                                color='#079967',
                                size=6,
                                opacity=0.85
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
                        yaxis_title='target',
                        legend=dict(
                            traceorder="normal"  # 图例顺序按添加顺序排列
                        )
                    )

                    # 更新布局
                    fig.update_layout(
                        title=dict(
                            text=f"[x_{x_eg_columns.index(x_i) + 1}] [P: {P}] [R: {R:.3f}]-[PrsR:{calculate_corr(y_exp_eg, y_obs_eg)[0]:.3f}]",
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

                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("")
                #
                # img_bytes = pio.to_image(fig, format="png", width=800, height=600, scale=2)
                #
                # # 下载按钮
                # st.download_button(
                #     label="📥 下载图像 (PNG)",
                #     data=img_bytes,
                #     file_name="plot.png",
                #     mime="image/png",
                #     use_container_width=True,
                # )
                #


# 7-6 Plot 区域 ---------------------------------------------------------------------------------------------------------





        # 7-6-1 拟合X与y_obs的 DPLS_obj  -----------------------------------------------------------------------------





# 8 预处理, 方法, 分类的参数控制面板 -----------------------------------------------------------------------------------------


st.markdown("")

# hr_second(dark_color="#244690", height=2, light_color="#26519d", left_px=0, right_px=0)
if use_data_type == use_data_options[1]:
    hr_second(dark_color="#244690", height=2, light_color="#26519d", left_px=3, right_px=3)
else:
    st.markdown("---")
st.markdown("")

# ⬅️ 使用三列布局，中间插入竖线
analys_panel_process, analys_panel_2, analys_panel_method, analys_panel_3, analys_panel_classify = st.columns(
    [1, 0.03, 1, 0.03, 1])

with analys_panel_process:
    st.markdown("<h3 style='text-align: center;'>预处理控制面板</h3>", unsafe_allow_html=True)
    st.markdown("")

    st.markdown("---", unsafe_allow_html=True)
    if preprocess_selection:
        with st.expander('pre-process', expanded=True):

            st.markdown('')
            preprocess_kwargs = param_controller(
                param_list=preprocess_selection,
                para_descriptions=preprocess_descriptions,
                param_controls=preprocess_param_controls,
                desc='预处理'
            )

        st.markdown('---')
    else:
        with st.expander("pre-process", expanded=False):
            st.markdown('---')

with analys_panel_method:
    st.markdown("<h3 style='text-align: center;'>方法控制面板</h3>", unsafe_allow_html=True)
    st.markdown("")

    st.markdown("---", unsafe_allow_html=True)
    if method_selection:
        with st.expander("methods", expanded=True):
            st.markdown('')
            method_kwargs = param_controller(
                param_list=method_selection,
                para_descriptions=method_descriptions,
                param_controls=method_param_controls,
                desc='方法'
            )
        st.markdown('---')
    else:
        with st.expander("methods", expanded=False):
            st.markdown('---')

with analys_panel_classify:
    st.markdown("<h3 style='text-align: center;'>分类控制面板</h3>", unsafe_allow_html=True)
    st.markdown("")

    st.markdown("---", unsafe_allow_html=True)
    if classify_selection:
        with st.expander("classifys", expanded=True):
            st.markdown('')
            classify_kwargs = param_controller(
                param_list=classify_selection,
                para_descriptions=classify_description,
                param_controls=classify_param_control,
                desc='分类'
            )
        st.markdown('---')
    else:
        with st.expander("classifys", expanded=False):
            st.markdown('---')

if preprocess_selection or method_selection or classify_selection:

    st.markdown("")
    st.markdown("")
    hr_second(dark_color="#244690", height=2, light_color="#26519d")

elif check_file_button:
    st.markdown("")
    st.markdown("")
    hr_second(dark_color="#244690", height=2, light_color="#26519d")

else:

    st.markdown("---")


# 9 在 detail 区打印参数 --------------------------------------------------------------------------------------------------


with host_panel_detail:

    detail_panel = st.columns([1, 1, 1, 1, 1])

    # 最先生成 Report 列
    with detail_panel[4]:

        # render_dataset_title('Reports', font_size=20)
        render_section_title('Reports', font_size=20)
        report_expander = st.expander('Reports', expanded=True)
        with report_expander:
            st.markdown('---')
            st.markdown('')

    #
    with detail_panel[0]:

        render_section_title('Samples', font_size=20)
        with st.expander('samples', expanded=True):
            st.markdown('---')
            display_detial_dict(file_param)
            st.markdown('')

    with detail_panel[1]:
        render_section_title('Pre process', font_size=20)
        with st.expander('pre-process', expanded=True):
            st.markdown('---')
            display_detial_dict(preprocess_print)
            st.markdown('')

    with detail_panel[2]:
        render_section_title('Methods', font_size=20)
        with st.expander('methods', expanded=True):
            st.markdown('---')
            display_detial_dict(method_print)
            st.markdown('')

    with detail_panel[3]:
        render_section_title('Classifiers', font_size=20)
        with st.expander('classifiers', expanded=True):
            st.markdown('---')
            display_detial_dict(classify_print)
            st.markdown('')

        def styled_f_call(c: str) -> str:
            return rf"\textcolor{{orange}}{{f}}\textcolor{{orange}}{{(}}\textcolor{{white}}{{{c}}}\textcolor{{orange}}{{)}}"



# 10 运行按钮前的 global 定义 ----------------------------------------------------------------------------------------------


# Methods 值的计算器
def cal_values(reverse=False):

    global use_files_dict, now_files_dict
    global total_mission, now_mission

    if reverse:
        direction = "BA"

    else:
        direction = "AB"

    # 默认第一列为原因
    if reverse != 1:
        reason = 0
        result = 1
    else:
        reason = 1
        result = 0

    per_db_mission = (len(preprocess_selection) + len(method_selection)*2)*len(use_files_dict)

    db_bar = st.progress(0, text=f"正在分析方向:{direction}")
    db_bar_progress = 0

    caling_files_dict = use_files_dict.copy()

    for db_name, db_data in caling_files_dict.items():

        pairs_processed = db_data['files_pair']

        for p , preprocess in enumerate(preprocess_selection):

            if preprocess in process:

                pairs_processed = parallel_wrapper(func=process[preprocess], file_value_dict=pairs_processed,
                                                   desc=preprocess, thread=Thread, reason=reason, result=result,
                                                   seed=seed_value, **preprocess_kwargs.get(preprocess, {}),
                                                   )
                now_mission += 1
                title_bar_progress.progress(now_mission / total_mission)

            db_bar_progress+=1
            db_bar.progress(db_bar_progress/per_db_mission)

        now_files_dict[db_name]['files_pair'].update(pairs_processed)

        df_list = []
        for m, method in enumerate(method_selection):

            if method in algorithms.keys():
                method_return = parallel_wrapper(func=algorithms[method], file_value_dict=pairs_processed, desc=method,
                                                 thread=Thread, reason=reason, result=result, seed=seed_value,
                                                 **method_kwargs.get(method, {}),
                                                 )
                method_return_DF = pd.DataFrame.from_dict(method_return, orient='index')
                named_method_DF = return_values_DF(method_return_DF, pre_process=preprocess_selection, method=method,
                                                   reverse=reverse, **method_kwargs.get(method, {}))
                df_list.append(named_method_DF)
            else:
                pass

            db_bar_progress += 2
            db_bar.progress(db_bar_progress/per_db_mission)

            now_mission += 2
            title_bar_progress.progress(now_mission / total_mission)

        try:

            result_df = pd.concat(df_list, axis=1)

        except ValueError:

            now_files_dict[db_name][f'{direction}_pair_methods'] = None

            pass

        else:
            now_files_dict[db_name][f'{direction}_pair_methods'] = result_df


# 分类方法执行器
def do_classify(classify_name, file_values:pd.DataFrame, train_list: list, test_list: list, **kwargs):

    global title_bar_progress
    global total_mission
    global now_mission

    if classify_name in classify_dict:
        classify_model = classify_dict[classify_name]

    else:
        return 0

    param_grid_dict = {
        "Logistic_classifyn": {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [100, 200, 1000],
            'penalty': ["l2", "l1", "elasticnet", "none"],
        },
        "Decision_Tree": {
            'max_depth': [3, 5, 10, None],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2,10],
        },
        "Random_Forest": {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 10],
            'criterion': ['gini', 'entropy'],
        },
        "SVM": {
            'C': [0.1, 2],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto'],
        },
        "KNN": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            "metric": ["euclidean", "manhattan"]
        },
        "Naive_Bayes": {},  # 没有超参数
        "LDA": {
            "solver": ["svd", "lsqr", "eigen"],
            "shrinkage": [None, "auto", "float"],

        }  # 没有超参数
    }

    param_grid = param_grid_dict.get(classify_name, {})

    for k, v in kwargs.items():
        param_grid.setdefault(k, v)

    if 'parameter_optimization' in param_grid:
        del param_grid['parameter_optimization']



    X = file_values.iloc[:, :-1]
    y = file_values.iloc[:, -1]

    try:

        X = to_2D_ary(X)
        y = np.array(y).flatten()

        accuracy_of_preds = []

        for train_idx, test_idx in zip(train_list, test_list):

            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]

            try:

                if kwargs['parameter_optimization']:

                    grid = GridSearchCV(classify_model, param_grid, cv=classify_cv, scoring='accuracy', n_jobs=Thread)
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

        now_mission += 1
        title_bar_progress.progress(now_mission/total_mission)

        return accuracy_of_preds

    except Exception as e:
        now_mission += 1
        st.error(e)


def del_cache():

    if 'X_eg' in st.session_state:
        del st.session_state['X_eg']

    if 'x_eg' in st.session_state:
        del st.session_state['x_eg']

    if 'x_picked_eg' in st.session_state:
        del st.session_state['x_picked_eg']

    if 'y_exp_eg' in st.session_state:
        del st.session_state['y_exp_eg']

    if  'eg_created' in st.session_state:
        del st.session_state['eg_created']

    if "create_eg_kwargs" in st.session_state:
        del st.session_state["create_eg_kwargs"]

    if "add_x_b" in st.session_state:
        del st.session_state["add_x_b"]
        del st.session_state["minus_x_b"]
        del st.session_state["add_xtox_b"]
        del st.session_state["minus_xtox_b"]

# 配置文件记录器
def save_config(config_folder_path: str, config_name: str):

    # 先删除缓存
    del_cache()

    if os.path.exists(config_folder_path):
        pass
    else:
        os.makedirs(config_folder_path)

    run_config = dict(st.session_state)

    #添加一些版本信息
    run_config['input_config_name'] = config_name
    run_config['Version'] = now_version

    save_config_path = os.path.join(config_folder_path, f'{config_name}.json')

    with open(save_config_path, 'w', encoding='utf-8') as f:
        json.dump(run_config, f, ensure_ascii=False, indent=4)


if save_config_button:

    if now_config_name:

        save_config(config_dir, f'{now_config_name}')
        with host_config_panel:
            st.info(f'已存为{now_config_name}.json')
    else:
        from datetime import datetime
        current_time = datetime.now().strftime("%H-%M")
        save_config(config_dir, f'{current_time}_run')
        with host_config_panel:
            st.info(f'已存为{current_time}_run.json')


with report_expander:

    if "input_config_name" in st.session_state:

        st.success(f"配置[{st.session_state['input_config_name']}]已载入")

if config_input:
    with report_expander:
        st.success(f"配置{selected_config}已载入")


def config_transfer():
    # 预处理流程
    preprocess_config = {
        f"预处理步骤: {step}": preprocess_kwargs[step] for idx, step in enumerate(st.session_state.get("preprocess_selection", []), start=1)
    }

    # 方法参数
    method_config = {
        f"分析方法: {method}": method_kwargs[method] for idx, method in enumerate(st.session_state.get("create_method_selection", []), start=1)
    }

    # 分类参数
    classify_config = {
        f"分类方法: {clf}": classify_kwargs[clf] for idx, clf in enumerate(st.session_state.get("classify_selection", []), start=1)
        if clf != "All"
    }

    # 合并所有参数
    all_config = {}
    all_config.update(file_param)
    all_config.update(preprocess_config)
    all_config.update(method_config)
    all_config.update(classify_config)

    st.info(all_config)


# 11 运行按钮  -----------------------------------------------------------------------------------------------------------


if run_button:

    st.sidebar.markdown("")

    # 点击分析后立即存储本次配置
    del_cache()
    save_config(config_dir, f'last_run_config')
    st.info("本次运行配置已存储")

    # 删除缓存

    now_files_dict = copy.deepcopy(use_files_dict)

    with title_bar:

        # 插入 CSS 控制进度条外边距
        st.markdown("""
            <style>
            div[data-testid="stProgress"] {
                margin-top: 0px;
            }
            </style>
        """, unsafe_allow_html=True)

        render_dataset_title("总进度", font_size=20)
        # 添加带标签的进度条（Streamlit >= 1.25）
        title_bar_progress = st.progress(0)

    with report_expander:
        st.info(f"Selected [{len(use_files_dict)}] Files, AB: [{total_1}], BA: [{total_0}]")

    try:

        # 分析方向控制

        run_AB = analys_direction in ["AB", "AB&BA"]
        run_BA = analys_direction in ["BA", "AB&BA"]

        # 数据选择

        all_results = {}

        if method_selection:

            total_mission = (run_BA + run_AB) * (2*len(method_selection) + len(preprocess_selection)) * len(use_files_dict) + len(classify_selection)
            method_results = []
            file_causes = []

            for db_dict in use_files_dict.copy().values():

                db_y = pd.DataFrame.from_dict(db_dict['files_cause'], orient='index')
                file_causes.append(db_y)

            y_df = pd.concat(file_causes, axis=0)

            if run_AB:

                st.success("正在分析方向:AB")

                cal_values()

                AB_db_methods_results = []

                for db_dict in now_files_dict.values():

                    AB_db_methods_results.append(db_dict['AB_pair_methods'])

                AB_methods_results = pd.concat(AB_db_methods_results, axis=0)
                method_results.append(AB_methods_results)


            if run_BA:

                st.success("正在分析方向:BA")

                cal_values(reverse=True)

                BA_db_methods_results = []

                for db_dict in now_files_dict.values():
                    BA_db_methods_results.append(db_dict['BA_pair_methods'])

                BA_methods_results = pd.concat(BA_db_methods_results, axis=0)
                method_results.append(BA_methods_results)

            # 文件处理

            # 汇总所有数据集, 所有方法的结果
            method_values_df = pd.concat(method_results, axis=1)

            st.success(f"Methods 分析完成!")
            st.dataframe(method_values_df)

            method_values_df_with_y = pd.concat([method_values_df, y_df], axis=1)

            all_results[f'Values-results_{file_direction}.xlsx'] = method_values_df_with_y

            if classify_selection:

                classify_results = {}
                num_rows_with_nan = method_values_df_with_y.isna().any(axis=1).sum()

                if num_rows_with_nan > 0:

                    if illegal_compatible_mode:
                        method_values_df_with_y = method_values_df_with_y.dropna(axis=0, how='any')
                        with report_expander:
                            st.warning(f"检测到[{num_rows_with_nan}]个非法样本, 已移除")
                        st.warning(f"检测到[{num_rows_with_nan}]个非法样本, 已移除")
                    else:
                        with report_expander:
                            st.warning(f"检测到[{num_rows_with_nan}]个非法样本, 若报错请尝试在[非法样本兼容模式]下分析")
                        st.warning(f"检测到[{num_rows_with_nan}]个非法样本, 若报错请尝试在[非法样本兼容模式]下分析")

                train_list, test_lit = spliter(method_values_df_with_y.shape[0],
                                               cv=classify_cv,
                                               mode=cv_mode,
                                               random_before=classify_shuffle_confirm,
                                               shuffle_seed=classify_shuffle_seed,
                                               )

                st.success('正在分类分析...')

                classify_bar = stqdm(classify_selection, total=len(classify_selection))
                for classify in classify_selection:
                    classify_bar.set_description(f"正在执行:{classify}")
                    classify_result = do_classify(classify, file_values=method_values_df_with_y,
                                                  train_list=train_list, test_list=test_lit, **classify_kwargs.get(classify, {}))
                    classify_results[classify] = classify_result
                    classify_bar.update(1)


                classify_results_df = pd.DataFrame.from_dict(classify_results, orient='index')
                classify_results_df['Mean'] = classify_results_df.mean(axis=1)
                st.success('Classify 分析完成!')

                # 完成后事项 ---------------------------------------------------------------------------------------------

                with report_expander:
                    st.success('Analysis completed')
                st.dataframe(classify_results_df)

                classify_file_name = f"Classify-results_{file_direction}"
                all_results[f"{classify_file_name}.xlsx"] = classify_results_df

            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for filename, df in all_results.items():
                    # 每个 DataFrame 转换为 Excel 文件流
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=True)
                    excel_buffer.seek(0)
                    # 将该 Excel 写入 zip 包
                    zip_file.writestr(filename, excel_buffer.read())

            # 重要：把 ZIP 指针移到开头
            zip_buffer.seek(0)

            # 下载按钮
            st.download_button(
                label="📦 下载所有结果 (ZIP)",
                data=zip_buffer,
                file_name=f"File_{Database_print}_{method_print}_{classify_print}_all.zip",
                mime="application/zip"
            )

            st.markdown('本次运行配置已保存, 点击下载按钮将重置页面')

        else:

            total_mission = (len(preprocess_selection)*len(use_files_dict)) if len(preprocess_selection) > 0 else 1
            cal_values()

        st.markdown('---')
        st.markdown("<h3 style='text-align: center;'>抽样检视面板</h3>", unsafe_allow_html=True)
        st.markdown('')

        # 定义运行完毕后的抽样检视面板, 只能再定义一次, 复用 expand_raw_now_files() 位置会不对 -------------------------------------

        with st.spinner('读取文件中...'):

            with st.expander('抽样检视面板'):

                st.markdown('')
                st.markdown('---')

                check_file_panel_raw, check_file_panel_sep2, check_file_panel_now, check_file_panel_3, check_file_panel_y = st.columns(
                    [1, 0.022, 1, 0.022, 0.6])

                with check_file_panel_raw:

                    with st.expander('原始值', expanded=True):
                        # 文件列表标题
                        cols = st.columns([3, 4])
                        cols[0].markdown("<div style='text-align: left; margin-top: 30px; padding-left:10px;'>文件名</div>",
                                         unsafe_allow_html=True)
                        cols[1].markdown(
                            f"<div style='text-align: right; margin-top: 20px; padding-right: 20px;'> <span style='color: #55dd99; font-size:20px;'><strong>{total_file}</strong></span> files</div>",
                            unsafe_allow_html=True)
                        st.markdown("---")

                        for db_name, db_values in use_files_dict.items():

                            db_len = len(db_values["files_pair"])

                            with st.expander(f'{db_name} | {db_len} files'):

                                for name, values in db_values["files_pair"].items():
                                    with st.expander(f'{name} | {values.shape}'):
                                        st.dataframe(values)

                    st.markdown("---")

                with check_file_panel_now:

                    with st.expander('Processed values', expanded=True):
                        # 文件列表标题
                        cols = st.columns([1.6, 4])
                        cols[0].markdown("<div style='text-align: left; margin-top: 30px; padding-left:10px;'>文件名</div>",
                                         unsafe_allow_html=True)

                        cols[1].markdown(
                            f"<div style='text-align: right; margin-top: 27px; padding-right: 20px;'> <span style='font-size:15px;'>{preprocess_print}</span></div>",
                            unsafe_allow_html=True)
                        st.markdown("---")

                        for db_name_, db_values_ in now_files_dict.items():

                            db_len = len(db_values_["files_pair"])

                            with st.expander(f'{db_name_} | {db_len} files'):

                                for name, values in db_values_["files_pair"].items():
                                    with st.expander(f'{name} | {values.shape}'):
                                        st.dataframe(values)

                    st.markdown("---")

                with check_file_panel_y:

                    with st.expander('Cause directions', expanded=True):
                        # 文件列表标题
                        cols = st.columns([4, 4])

                        cols[0].markdown(
                            f"<div style='text-align: left; margin-top: 20px; padding-left: 20px;'> <span style='color: #4477dd; font-size:20px;'><strong>[{total_1}] </strong></span> A -> B</div>",
                            unsafe_allow_html=True)

                        cols[1].markdown(
                            f"<div style='text-align: right; margin-top: 20px; padding-right: 20px;'> <span style='color: #dd4477; font-size:20px;'><strong>[{total_0}] </strong></span> B -> A</div>",
                            unsafe_allow_html=True)
                        st.markdown("---")

                        for db_name, db_values in use_files_dict.items():
                            db_len = len(db_values["files_cause"])

                            with st.expander(f'{db_name} | {db_len} files'):
                                files_cause_df = pd.DataFrame.from_dict(db_values["files_cause"], orient='index')

                                st.dataframe(files_cause_df)

                    st.markdown("---")

        # base_dir = os.path.dirname(os.path.abspath(__file__))
        # temp_now_file_path = os.path.join(base_dir, 'temp_now_file')
        #
        # if os.path.exists(temp_now_file_path):
        #     pass
        # else:
        #     os.makedirs(temp_now_file_path)
        #
        # json_ready_dict = {
        #     key: df.to_dict(orient="records")  # 每个 DataFrame 转字典
        #     for key, df in now_files_dict.items()
        # }
        #
        # with open(rf'{temp_now_file_path}\last_now_file_record.json', 'w', encoding='utf-8') as f:
        #     json.dump(json_ready_dict, f, ensure_ascii=False, indent=4)


    except Exception as e:

        with report_expander:
            st.error("❌ 分析出错, 详情见底部报告")

        st.error(e)
        msg = traceback.print_exc()
        print(msg)



# 12 文件检视实现 ---------------------------------------------------------------------------------------------------------


elif check_file_button:

    def expand_raw_now_files(raw: dict, total_0, total_1, expand=True):

        check_file_panel = st.columns([1, 0.03, 1, 0.03, .72])

        with check_file_panel[0]:

            with st.expander('原始值', expanded=expand):
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

                    db_len = len(db_values["files_pair"])

                    with st.expander(f'{db_name} | {db_len} files'):

                        for name, values in db_values["files_pair"].items():
                            with st.expander(f'{name} | {values.shape}'):
                                st.dataframe(values)

            st.markdown("---")

        with check_file_panel[2]:

            with st.expander('Cause directions', expanded=expand):
                # 文件列表标题
                raw_y_cols = st.columns([4, 4])

                raw_y_cols[0].markdown(
                    f"<div style='text-align: left; margin-top: 20px; padding-left: 20px;'> <span style='color: #4477dd; font-size:20px;'><strong>[{total_1}] </strong></span> A -> B</div>",
                    unsafe_allow_html=True)

                raw_y_cols[1].markdown(
                    f"<div style='text-align: right; margin-top: 20px; padding-right: 20px;'> <span style='color: #dd4477; font-size:20px;'><strong>[{total_0}] </strong></span> B -> A</div>",
                    unsafe_allow_html=True)
                st.markdown("---")

                for db_name, db_values in raw.items():
                    db_len = len(db_values["files_cause"])

                    with st.expander(f'{db_name} | {db_len} files'):
                        files_cause_df = pd.DataFrame.from_dict(db_values["files_cause"], orient='index')

                        st.dataframe(files_cause_df)

            st.markdown("---")
            time.sleep(1)

        with check_file_panel[4]:

            with st.expander('Plots', expanded=expand):

                raw_p_cols = st.columns([4, 4])

                raw_p_cols[0].markdown(
                    "<div style='text-align: left; margin-top: 30px; padding-left:10px;'>图像数量</div>",
                    unsafe_allow_html=True)

                raw_p_cols[1].markdown(
                    f"<div style='text-align: right; margin-top: 23px; padding-right: 20px;'> <span style='color: #55dd99; font-size:20px;'><strong>{total_file}</strong></span> plots</div>",
                    unsafe_allow_html=True)

                st.markdown("---")

                for db_name, db_values in raw.items():

                    db_len = len(db_values["files_pair"])

                    with st.expander(f'{db_name} | {db_len} files'):

                        if print_pred:

                            checking_preds = parallel_wrapper(cal_DPLS_pred, db_values["files_pair"], desc="cal_DPLS_pred", reason=0, result=1, thread=Thread)

                        for name, values in db_values["files_pair"].items():

                            with st.expander(f'{name} | {values.shape}'):

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

                                    checking_pred = checking_preds[name][1]

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
                                        text=f"DPLSR2: {checking_preds[name][0]:.3f}" if print_pred else f"",
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

            st.markdown("---")
            time.sleep(1)

    st.markdown("")
    st.markdown("")
    st.markdown("<h3 style='text-align: center;'>文件检视面板</h3>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("---")

    with st.spinner("正在加载文件..."):

        expand_raw_now_files(use_files_dict, total_0=total_0, total_1=total_1, expand=True)

# 未运行时的底部逻辑
else:
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown("参数设置完毕后即可点击“开始分析”。")

# 13 签名 ---------------------------------------------------------------------------------------------------------------

st.markdown("""
    <style>
    .footer-text {
        position: fixed;
        bottom: 10px;
        right: 15px;
        font-size: 0.75rem;
        color: #a0a0a0;
        z-index: 999;
    }
    </style>
    <div class="footer-text">
        Author: Siwei Jiang, Zheming Yuan ｜ Organization: College of Plant Protection,
         Hunan Agricultural University
    </div>
""", unsafe_allow_html=True)

# 14 默认设置 ------------------------------------------------------------------------------------------------------------

