import streamlit as st
import pandas as pd
import numpy as np
from typing import Literal, Union

from custom_html_module import *
import time
from cause_pair_functions.muti_func_test import gen_y_exp
from cause_pair_functions.DPLS_jj import DPLS
import copy
import plotly.graph_objects as go


# 定义区 ----------------------------------------------------------------------------------------------------------------

st.set_page_config(layout="wide")

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



# 7-2-2 与 本地文件模式共用的功能部分 --------------------------------------------------------------------------------

# 模拟样本的独立功能区, 单独于 host_panel_setting 外 -------------------------------------------------------------------------


st.markdown('')
st.markdown('')
st.markdown('')
hr_second(dark_color="#244690", height=2, light_color="#26519d")
st.markdown('')
st.markdown('')
st.markdown('')

# 7-2-3 定义生成模拟数据的控制面板 ---------------------------------------------------------------------------------------

create_control_panel, create_sep1, create_preview_panel = st.columns([.382, 0.02, .618])

st.markdown('')
st.markdown('')

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

    formular_refresh_col, formular_sep_1, formular_title_add_x, add_x_col, minus_x_col,  formular_sep_2, formular_title_add_xtox, add_xtox_col, minus_xtox_col, formular_sep_3  = st.columns([.6, .3, .75, .21, .21, .3, .75, .21, .21, 3])

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

    create_main_sets_col_1, create_main_sep, create_main_sets_col_2 = st.columns([.5,.02,.5])

    # 7-2-3-1 无关功能 -------------------------------------------------------------------------------------------

    with create_main_sets_col_1:

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

            with create_redun_col:

                create_redun = st.selectbox("**开启冗余** ⚠️", [False, True], key="create_redun",
                                            help=create_help_dict.get("create_redun", "无描述"),)


    # 7-2-3-3互作功能 --------------------------------------------------------------------------------------------

    with create_main_sets_col_2:

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


            with create_xseed_col:

                create_xseed = st.number_input("x的随机种子", min_value=0, max_value=20000, step=1, value=409, key="create_xseed",
                                               help="输入x数量后, 每个x会在你给出的定义域上随机抽取n个(每个x抽到的不一样, 即使你只给了一个随机种子), x的随机种子用于决定这个抽取过程"
                                               )

    create_advanced_sets_expander = st.expander("高级设置")

    with create_advanced_sets_expander:

        # 7-2-3-5 一些其他功能 ----------------------------------------------------------------------------------------


        st.markdown("")
        create_linear_coef_col, create_linear_intercept_col = st.columns([1, 1])

        with create_linear_coef_col:

            create_linear_coef_range = st.slider("线性函数系数范围", -10, 10, (-3, 3), key="create_linear_coef_range", step=1)

            create_redun_count = st.slider("X的冗余最大层 ⚠️", min_value=1, max_value=8, step=1, value=5,
                                           key="create_redun_count",
                                           help="[⛓️‍存在依赖项]: 此项发挥作用需要 [开启冗余] 被设置为 [True],  \n"
                                                "[⚠️激进的参数]: 此参数将显著影响函数复杂度,  \n"
                                                "[互作层级]: 指原变量外嵌套的 f 层级, 如 x_3 = sin(x_2), 此时 [嵌套层级] = 1  \n"
                                                " x_3 = cos(sin(x_2))^2, 此时 [嵌套层级] = 3  \n"
                                                "[互作层级] 越高,则越可能生成复杂的 [冗余项]  \n"
                                           )


            create_x_mode = st.selectbox("X取值模式", ["均匀", "成长型", "抛物线型"], key="create_x_mode",
                                        help="[均匀]模式时,X在定义域内取到每个点的概率相同, [成长型]:X越大取到的概率越大, [抛物线型]:中间的X被取到的概率更大")

            create_x_mode_transfer = {
                "均匀":"uniform",
                "成长型":"grow",
                "抛物线型":'parabola'
            }

            create_x_mode = create_x_mode_transfer[create_x_mode]


        with create_linear_intercept_col:

            create_linear_intercept_range = st.slider("线性函数截距范围", -25, 25, (-5, 5), key='create_linear_intercept_range', step=2)

            create_redun_slope = st.slider("X的冗余倾向 ⚠️", min_value=1.0, max_value=8.0, step=0.5, value=2.5,
                                           key="create_redun_slope",
                                           help="[⛓️‍存在依赖项]: 此项发挥作用需要 [开启冗余] 被设置为 [True],  \n"
                                                "[⚠️激进的参数]: 此参数将显著影响函数复杂度,  \n"
                                                "[冗余倾向] 越高,则越可能生成复杂的 [冗余项], 也越可能使用 [冗余特征] 参与构成 [y]  \n"
                                           )

            create_redun_seed = st.number_input("冗余随机种子", min_value=0, max_value=20000, step=1, value=73, key="create_redun_seed",
                                               help="开启冗余后, 随机种子用于决定冗余的状态"
                                               )

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

    check_file_button = st.button("**检视**", use_container_width=True, icon="🔍")
    hr_second(dark_color="#244690", height=2.5, light_color="#26519d")

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

# 7-4 定义模拟数据检视逻辑 ---------------------------------------------------------------------------------------------

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


    # 7-5-4 打印公式的latex格式 ---------------------------------------------------------------------------------------

    latex_expr = apply_colored_brackets(formula_eg)
    st.latex(latex_expr)
    st.markdown('')
    st.markdown("")
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown('')

    st.markdown("---")

# 独立与互作特征鉴别器(待升级)
def count_unique_ai(expr: str, list_a: list) -> int:
    count = 0
    for item in list_a:
        if item in expr:
            count += 1
    return count


# 读取颜色（默认浅色）

with create_control_panel:

    formula_panel1_col1, f_sep1, formula_panel1_col2 = st.columns([.5, 0.02, .5])

    with formula_panel1_col1:

        create_method_expander = st.expander("选择分析的方法", expanded=True)

        with create_method_expander:

            # 4-3 选择方法
            create_method_selection = st.selectbox(
                label="分析方法（可多选）", key="create_method_selection",
                options="DPLSR", label_visibility="collapsed",
            )

            # 4-4 方法的打印内容
            if create_method_selection:
                method_print = {m + 1: method_ for m, method_ in enumerate(create_method_selection)}
            else:
                method_print = {}

        # 使用 expander 创建一个可折叠/展开的区域
        DPLSR_param_dict_ = DPLSR_param_dict.copy()
        DPLS_kwargs = param_controller(
            param_list=['DPLSR'],
            para_descriptions={'DPLSR': method_descriptions.get('DPLSR')},
            param_controls={"DPLSR": DPLSR_param_dict_},
            desc='方法',
            a_copied_dict=True
        )

        if not DPLS_kwargs['DPLSR']["distance_pattern"]:
            DPLS_kwargs['DPLSR']["distance_pattern"] = ['Euc']


    with formula_panel1_col2:

        create_exe_expander_2 = st.expander(f"**输出**", expanded=True)

        # 7-5-6-1 打印模拟样本的生成参数  -------------------------------------------------------------------------------

        with create_exe_expander_2:
            st.markdown('')
            st.markdown('')
            st.markdown("---")

            illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False], key="illegal_compatible_mode")

            direction_panel = st.columns([1, 1])

            # 文件方向
            with direction_panel[0]:
                file_direction = st.selectbox("样本方向", ["X - y", "y - X", "X - y&y - X"], key="file_direction")

            # 分析方向
            with direction_panel[1]:
                analys_direction = st.selectbox("属性方向", ["X - y", "y - X", "X - y&y - X"], key="analys_direction")

            col_file_num, col_test_seed = st.columns([1, 1])

            with col_file_num:
                test_files_num = st.number_input(
                    "生成类似函数的个数",
                    min_value=10,
                    max_value=10000,
                    value=50,
                    step=10
                    , key="test_files_num")

            with col_test_seed:
                seed_value = st.number_input(
                    "类似函数抽样种子",
                    min_value=0,
                    max_value=20000,
                    value=42,
                    step=1
                    , key="seed_value")

            thresh_range = st.slider("样本数限制在", 0, 10000, (100, 1500), key="thresh_range", step=100)
            st.markdown("")

            run_button = st.button("run_button")

created_data = {}
total_0 = 0
total_1 = 0
total_file=0

# if check_file_button or run_button:
#
#     with st.spinner('正在生模拟数据...'):
#
#         create_times = test_files_num // (create_use_x_num + create_xtox_num)
#
#         create_func_seeds = gen_seed(create_times, rand_seed=create_funcseed, gen_times=1)[0]
#         create_x_seeds = gen_seed(create_times, rand_seed=create_xseed, gen_times=1)[0]
#
#         for create_i in range(create_times):
#
#             np.random.seed(create_func_seeds[create_i])
#
#             create_kwargs["func_seed"] = create_func_seeds[create_i]
#             create_kwargs["x_seed"] = create_x_seeds[create_i]
#
#             eg_create_samples = np.random.randint(thresh_range[0], thresh_range[1])
#
#             # 7-4-1 参数选择完毕, 开始生成模拟数据 ----------------------------------------------------------------------
#
#             time.sleep(1)
#
#             x_create, X_create, y_exp_create, x_picked_create = gen_y_exp(sample_num=eg_create_samples,
#                                                                           **create_kwargs)
#             np.random.seed(create_x_seeds[create_i])
#             y_create_obs = y_exp_create + np.random.normal(size=y_exp_create.shape[0], loc=0, scale=y_exp_create.std() * create_noise)
#
#             create_func_name = 'y=' + '+'.join(list(X_create))
#
#             if use_x_piked:
#
#                 create_i_use = x_create.copy()[x_picked_create]
#
#             else:
#                 create_i_use = x_create.copy()
#
#             create_i_use["y"] = y_create_obs
#             created_data_pair, created_pair_name, created_data_cause = return_cause_pair(create_i_use, relation=file_direction, prefix=f'[{create_func_name}]')
#
#             created_data[create_func_name] = {"files_pair": dict(zip(created_pair_name, created_data_pair)),
#                                        "files_cause": dict(zip(created_pair_name, created_data_cause))}
#
#             count_0 = created_data_cause.count(0)
#             count_1 = created_data_cause.count(1)
#
#             total_0 += count_0
#             total_1 += count_1
#             total_file += len(created_data_pair)
#
#     # 7-4-2 模拟数据存入全局变量: use_files_dict -----------------------------------------------------------------------
#
#     use_files_dict = created_data
#
#     # 传给 detail_panel 的参数
#     file_param = {
#         "正在分析": "模拟样本",
#         "非法样本兼容": illegal_compatible_mode,
#         '样本方向': file_direction,
#         '分析方向': analys_direction,
#         '文件抽样数': test_files_num,
#         '文件抽样种子': seed_value,
#         '样本量': thresh_range,
#         "开启冗余": f"是" if create_redun else "否",
#         "仅生成线性样本": create_linear_limit,
#     }

    # 7-5-5 打印样本的 X 与 y -----------------------------------------------------------------------------------------

    # 7-5-6 打印模拟样本的生成参数 与 参数符合度的检测结果 ------------------------------------------------------------------

with create_preview_panel:
    st.markdown("")

    plot_sep_0, plot_P_col, plot_sep_1,plot_addnoise_col, plot_sep_4, plot_y_exp_col, plot_sep_2, plot_y_obs_col, plot_sep_3 = st.columns(
        [.2, 1.3,.2,1.3,.2,.5,.2, .5, 3.5])


    with plot_P_col:

        enforce_P = st.slider("硬定位的 P: ", min_value=-1, max_value=DPLS_kwargs['DPLSR']['max_iter'] - 1,
                              value=-1, key='enforce_P')

    with plot_addnoise_col:

        create_noise = st.slider("y_obs 噪音强度", min_value=0.0, max_value=5.0, step=0.1, value=0.5,
                                 key="create_noise",
                                 help="生成模拟样本后往[期望值y_exp]添加的噪音强度"
                                 )
        y_obs_eg = y_exp_eg + np.random.normal(size=y_exp_eg.shape[0], loc=0,
                                               scale=y_exp_eg.std() * create_noise)
        y_df_eg = pd.DataFrame()
        y_df_eg["y_exp"] = y_exp_eg
        y_df_eg["y_obs"] = y_obs_eg

    with plot_y_exp_col:
        st.markdown("")
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        print_exp = st.checkbox('y_exp', value=True, key='print_exp')


    with plot_y_obs_col:
        st.markdown("")
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        print_obs = st.checkbox('y_obs', value=False, key='print_obs')

    st.markdown("")
    st.markdown("")
    st.markdown("")

    with st.spinner("正在拟合..."):

        pred_obj = DPLS(**DPLS_kwargs["DPLSR"]).fit(x_eg_use.copy(), y_df_eg['y_obs'].copy(),
                                                    **DPLS_kwargs["DPLSR"])

    cols = st.columns(2)  # 定义两列

    for idx, x_i in enumerate(x_picked_eg):

        x = list(x_eg_use.columns).index(x_i)

        if enforce_P == -1:

            if DPLS_kwargs["DPLSR"]["R_mode"] == 'single':

                y_pred = pred_obj.y_pred[x]
                print("y_pred:", y_pred.shape)

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
                text=f"[x_{x_eg_columns.index(x_i) + 1}] [P: {P}] [R: {R:.2f}]-[PrsR:{calculate_corr(y_exp_eg, y_obs_eg)[0]:.2f}]",
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

create_check_file_expander = st.expander("检视预览函数样本", expanded=False)
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

hr_second(dark_color="#244690", height=2.5, light_color="#26519d")