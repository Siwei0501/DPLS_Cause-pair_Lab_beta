import streamlit as st
import pandas as pd
import numpy as np
from typing import Literal, Union


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


# 参数收集器
def param_controller(param_list: list, para_descriptions: dict, param_controls: dict, desc='', a_copied_dict=False) -> dict:
    param_kwargs = {}

    m = 0
    for param in param_list:

        if param in param_controls:

            with st.expander(
                    f"{desc}\t{chr(9312 + m)}&nbsp;&nbsp;{param}:&nbsp;&nbsp;&nbsp;{para_descriptions.get(param, '')} ",
                    expanded=True):
                param_kwargs[param] = {}
                st.markdown('---')

                for param_key, control in param_controls[param].items():

                    if a_copied_dict:
                        key_name = f"{param}_{param_key}_"

                    else:
                        key_name = f"{param}_{param_key}"

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
        pair_name.extend([relation_ + f"{prefix}[{str(col_name)}]" for col_name in not_pair_data.columns[:-1]])

        if relation_ == "AB":

            pair_cause.extend([1] * len(data_in_pair))


        else:
            pair_cause.extend([0] * len(data_in_pair))

    return pair_data, pair_name, pair_cause

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
