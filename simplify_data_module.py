import streamlit as st
import pandas as pd
import numpy as np
from typing import Literal, Union

from custom_html_module import *
import io
import zipfile
import time
from cause_pair_functions.muti_func_test import gen_y_exp
from cause_pair_functions.DPLS_jj import DPLS
import copy
import plotly.graph_objects as go
import plotly.io as pio
import random


# 定义区 ----------------------------------------------------------------------------------------------------------------

st.set_page_config(layout="wide")


@st.cache_data(show_spinner=False)
def export_multiple_plots_to_zip(fig_dict: dict[str, go.Figure]) -> bytes:
    """
    将多个 Plotly 图像导出为 PNG，并打包为 ZIP。

    参数:
        fig_dict: 一个字典，键为文件名，值为 plotly 图对象

    返回:
        bytes: zip 文件的字节内容
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, fig in fig_dict.items():
            img_bytes = pio.to_image(fig, format="png", width=800, height=600, scale=2)
            zip_file.writestr(f"{filename}.png", img_bytes)

    zip_buffer.seek(0)
    return zip_buffer.read()

# 7-2-2 与 本地文件模式共用的功能部分 --------------------------------------------------------------------------------

# 模拟样本的独立功能区, 单独于 host_panel_check_file 外 -------------------------------------------------------------------------

def simplify_data_module(block_id, thread=1):
    
    hr_second(dark_color="#244690", height=2, light_color="#26519d")
    
    # 7-2-3 定义生成模拟数据的控制面板 ---------------------------------------------------------------------------------------
    if st.session_state.get(f"{block_id}_use_files_dict", False):

        use_files_dict = st.session_state.get(f"{block_id}_use_files_dict")

    else:

        use_files_dict = {}

    create_control_panel, create_sep1, create_preview_panel = st.columns([.382, 0.026, .618])
    
    with create_sep1:
        st.markdown("""
            <style>
                .vertical-line {
                    width: 1px;
                    height: 1722px;
                    margin: auto;
                    margin-top: -5px;
                }
    
                @media (prefers-color-scheme: dark) {
                    .vertical-line {
                        background-color: #333333;
                    }
                }
    
                @media (prefers-color-scheme: light) {
                    .vertical-line {
                        background-color: #d8d8d8;
                    }
                }
            </style>
    
            <div class="vertical-line"></div>
        """, unsafe_allow_html=True)
    
    with create_control_panel:
        st.markdown('')
        st.markdown('')
        create_control_sep0, create_func_control_panel_expander = st.columns([.01, 1])
        create_control_sep1, create_output_control_panel_ = st.columns([.01, 1])

        with create_func_control_panel_expander:
    
            st.markdown(
                "<h3 style='text-align: center;'>函数形态控制面板</h3>",
                unsafe_allow_html=True
            )
    
            st.markdown('---')
            st.markdown("")
            # create_func_control_panel_expander = st.expander("DPLS_Lab", expanded=True)

    st.markdown('')
    st.markdown('')
    
    with create_preview_panel:
        
        st.markdown('')
        st.markdown('')
        st.subheader('函数')
        st.markdown('---')
        st.markdown("")
    
        # 7-5-1 定义内容区 -----------------------------------------------------------------------------------------------
    
        formular_refresh_col, formular_sep_1, formular_title_add_x, add_x_col, minus_x_col,  formular_sep_2, formular_title_add_xtox, add_xtox_col, minus_xtox_col, formular_sep_3  = st.columns([.6, .3, .75, .21, .21, .3, .75, .21, .21, 2.1])
    
        # with formular_title_col:
        #     render_section_title("你的函数 be like :", underline=False)
    
        with formular_refresh_col:
    
            f_refresh_button = st.button("Clear", use_container_width=True, key=f"{block_id}_clear_button")
            if f_refresh_button:
    
                st.session_state[f"{block_id}_formular_refresh"]= True
    
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
    
        st.session_state[f'{block_id}_add_x'] = False
        st.session_state[f'{block_id}_add_xtox'] = False
        st.session_state[f'{block_id}_minus_x'] = False
        st.session_state[f'{block_id}_minus_xtox'] = False
    
        with add_x_col:
            add_x = st.button("＋", key=f"{block_id}_add_x_b")
            if add_x:
                st.session_state[f"{block_id}_add_x"] = True
    
        with minus_x_col:
            minus_x = st.button("－", key=f"{block_id}minus_x_b")
            if minus_x:
                st.session_state[f"{block_id}_minus_x"] = True
    
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
            add_xtox = st.button("＋", key=f"{block_id}_add_xtox_b")
            if add_xtox:
                st.session_state[f"{block_id}_add_xtox"] = True
        with minus_xtox_col:
            minus_xtox = st.button("－", key=f"{block_id}_minus_xtox_b")
            if minus_xtox:
                st.session_state[f"{block_id}_minus_xtox"] = True

        with formular_sep_3:

            formular_sep_3_col1, formular_sep_3_col2, formular_sep_3_col3 = st.columns([.5, 1, .18])
            with formular_sep_3_col2:
                formular_name = st.text_input("存储此函数", placeholder="存储此配置", label_visibility="collapsed", key=f"{block_id}_formular_eg_name")
            with formular_sep_3_col3:
                st.button("⏬", key=f"{block_id}_formular_eg_name_button", use_container_width=True)
    
    with create_func_control_panel_expander:

    
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
    
        create_main_sets_col_1, create_main_sets_col_2 = st.columns([.5,.5])
    
        # 7-2-3-1 无关功能 -------------------------------------------------------------------------------------------
    
        with create_main_sets_col_1:
    
            unrelated_expander = st.expander("**无关**", expanded=True)
    
            with unrelated_expander:
    
                create_use_x_col, create_x_col = st.columns([1, 1])
    
                with create_x_col:
    
                    create_interact_num = st.number_input("无关的X数量", min_value=0, max_value=20, step=1, value=0,
                                                       key=f"{block_id}_create_param_num",
                                                       help=create_help_dict.get("create_param_num","无描述")
                                                       )
    
                with create_use_x_col:
    
                    if f"{block_id}_create_use_x_num" in st.session_state:

                        if st.session_state.get(f"{block_id}_add_x", False) or st.session_state.get(f"{block_id}_add_xtox", False):
    
                            create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                               max_value=20,
                                                               step=1, value=st.session_state[f"{block_id}_create_use_x_num"]+1, key=f"{block_id}_create_use_x_num",
                                                               help=create_help_dict.get("create_use_x_num", "无描述"),
                                                               )
    
                        elif st.session_state.get(f"{block_id}_minus_x", False) or st.session_state.get(f"{block_id}_minus_xtox", False):
    
                            if st.session_state[f"{block_id}_create_use_x_num"] > 1:
                                create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                                   max_value=20,
                                                                   step=1, value=st.session_state[f"{block_id}_create_use_x_num"]-1, key=f"{block_id}_create_use_x_num",
                                                                   help=create_help_dict.get("create_use_x_num", "无描述"),
                                                                   )
                            else:
    
                                create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                                   max_value=20,
                                                                   step=1, value=1, key=f"{block_id}_create_use_x_num",
                                                                   help=create_help_dict.get(f"create_use_x_num", "无描述"),
                                                                   )
    
                        else:
    
                            create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                               max_value=20,
                                                               step=1, value=st.session_state[f"{block_id}_create_use_x_num"], key=f"{block_id}_create_use_x_num",
                                                               help=create_help_dict.get("create_use_x_num", "无描述"),
                                                               )

    
                    else:
    
                        create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                           max_value=20,
                                                           step=1, value=2, key=f"{block_id}_create_use_x_num",
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
    
                    if f"{block_id}_create_x_num" in st.session_state:
    
                        if st.session_state.get(f"{block_id}_add_x", False):
    
                            create_x_num = st.number_input("独立项数量", min_value=create_x_limit, max_value=10, step=1,
                                                              value=st.session_state[f"{block_id}_create_x_num"] + 1,
                                                              key=f"{block_id}_create_x_num",
                                                              help=create_help_dict.get("create_x_num", "无描述"),
                                                              )
    
                        elif st.session_state.get(f"{block_id}_minus_x", False) and st.session_state[f"{block_id}_create_x_num"] > create_x_limit:
    
                            create_x_num = st.number_input("独立项数量", min_value=create_x_limit, max_value=10, step=1,
                                                              value=st.session_state[f"{block_id}_create_x_num"] - 1,
                                                              key=f"{block_id}_create_x_num",
                                                              help=create_help_dict.get("create_x_num", "无描述"),
                                                              )
    
                        else:
    
                            create_x_num = st.number_input("独立项数量", min_value=create_x_limit, max_value=10, step=1,
                                                              value=st.session_state[f"{block_id}_create_x_num"],
                                                              key=f"{block_id}_create_x_num",
                                                              help=create_help_dict.get("create_x_num", "无描述"),
                                                              )
    
    
                    else:
    
                        create_x_num = st.number_input("独立项数量", min_value=0, max_value=20, step=1,
                                                       value=2,
                                                       key=f"{block_id}_create_x_num",
                                                       help=create_help_dict["create_x_num"],
                                                       )
    
                with create_redun_col:
    
                    create_redun = st.selectbox("**开启冗余** ⚠️", [False, True], key=f"{block_id}_create_redun",
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
    
    
                    if f"{block_id}_create_xtox_num" in st.session_state:
    
    
                        if st.session_state.get(f"{block_id}_add_xtox", False):
    
                            create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10, step=1,
                                                              value=st.session_state[f"{block_id}_create_xtox_num"] + 1,
                                                              key=f"{block_id}_create_xtox_num",
                                                              help=create_help_dict.get("create_xtox_num", "无描述"),
                                                              )
    
                        elif st.session_state.get(f"{block_id}_minus_xtox", False) and st.session_state[f"{block_id}_create_xtox_num"] > create_xtox_limit:
    
                            create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10, step=1,
                                                              value=st.session_state[f"{block_id}_create_xtox_num"] - 1,
                                                              key=f"{block_id}_create_xtox_num",
                                                              help=create_help_dict.get("create_xtox_num", "无描述"),
                                                              )
    
                        else:
    
                            create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10, step=1,
    
                                                              value=st.session_state[f"{block_id}_create_xtox_num"],
                                                              key=f"{block_id}_create_xtox_num",
                                                              help=create_help_dict.get("create_xtox_num", "无描述"),
                                                              )
    
                    else:
                        create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10, step=0, value=0,
                                                          key=f"{block_id}_create_xtox_num",
                                                          help=create_help_dict.get("create_xtox_num", "无描述"),
                                                          )
    
                with create_xtox_level_col:
    
                    create_xtox_level = st.number_input("互作最高参与项", min_value=2, max_value=5, step=1, value=2,
                                                        key=f"{block_id}_create_xtox_level",
                                                        help=create_help_dict.get("create_xtox_level","无描述"),
                                                        )
    
            # 7-2-3-4 随机种子 -------------------------------------------------------------------------------------------
    
            creation_seed_expander = st.expander("**随机种子**", expanded=True)
    
            with creation_seed_expander:
    
                create_funcseed_col, create_xseed_col = st.columns([1, 1])
    
    
                with create_funcseed_col:
    
                    if f"{block_id}_formular_refresh" in st.session_state and "create_funcseed" in st.session_state:
    
                        if st.session_state[f"{block_id}_formular_refresh"]:
    
                            create_funcseed = st.number_input("f(x)随机种子", min_value=0, max_value=20000, step=1, value=st.session_state[f"{block_id}_create_funcseed"]+1, key=f"{block_id}_create_funcseed",
                                                           help=create_help_dict.get("create_funcseed", "无描述"),
                                                           )
    
                        else:
    
                            create_funcseed = st.number_input("f(x)随机种子", min_value=0, max_value=20000, step=1,
                                                              value=st.session_state[f"{block_id}_create_funcseed"],
                                                              key=f"{block_id}_create_funcseed",
                                                              help=create_help_dict.get("create_funcseed", "无描述"),
                                                              )
    
                        st.session_state[f"{block_id}_formular_refresh"] = False
    
    
                    else:
    
                        create_funcseed = st.number_input("f(x)随机种子", min_value=0, max_value=20000, step=1, value=73, key=f"{block_id}_create_funcseed",
                                                       help=create_help_dict.get("create_funcseed", "无描述"),
                                                       )
                        st.session_state[f"{block_id}_formular_refresh"] = False
    
    
                with create_xseed_col:
    
                    create_xseed = st.number_input("x的随机种子", min_value=0, max_value=20000, step=1, value=409, key=f"{block_id}_create_xseed",
                                                   help="输入x数量后, 每个x会在你给出的定义域上随机抽取n个(每个x抽到的不一样, 即使你只给了一个随机种子), x的随机种子用于决定这个抽取过程"
                                                   )

        with st.expander("", expanded=True):

            create_x_bank = st.multiselect(label="**可出现的f(x):**", options=["All"] + list(function_dict.keys()),
                                           default=["线性函数", "正弦函数", "二次函数"],
                                           key=f"{block_id}_create_x_bank",
                                           help=create_help_dict.get("create_x_bank", "无描述")
                                           )

            if f"{block_id}_All" in create_x_bank:
                create_x_bank = list(function_dict.keys())

            if not create_x_bank:
                create_x_bank = ["正弦函数"]


            create_define = st.slider("定义域", -5.0, 5.0, (-1.0, 1.0), key=f"{block_id}_create_define", step=0.1)

            create_define_left = create_define[0]
            create_define_right = create_define[1]

        create_advanced_sets_expander = st.expander("**高级设置**")
    
        with create_advanced_sets_expander:
    
            # 7-2-3-5 一些其他功能 ----------------------------------------------------------------------------------------
    
    
            st.markdown("")
            create_linear_coef_col, create_linear_intercept_col = st.columns([1, 1])
    
            with create_linear_coef_col:
    
                create_linear_coef_range = st.slider("线性函数系数范围", -10, 10, (-3, 3), key=f"{block_id}_create_linear_coef_range", step=1)
    
                create_redun_count = st.slider("X的冗余最大层 ⚠️", min_value=1, max_value=8, step=1, value=5,
                                               key=f"{block_id}_create_redun_count",
                                               help="[⛓️‍存在依赖项]: 此项发挥作用需要 [开启冗余] 被设置为 [True],  \n"
                                                    "[⚠️激进的参数]: 此参数将显著影响函数复杂度,  \n"
                                                    "[互作层级]: 指原变量外嵌套的 f 层级, 如 x_3 = sin(x_2), 此时 [嵌套层级] = 1  \n"
                                                    " x_3 = cos(sin(x_2))^2, 此时 [嵌套层级] = 3  \n"
                                                    "[互作层级] 越高,则越可能生成复杂的 [冗余项]  \n"
                                               )
    
    
                create_x_mode = st.selectbox("X取值模式", ["均匀", "成长型", "抛物线型"], key=f"{block_id}_create_x_mode",
                                            help="[均匀]模式时,X在定义域内取到每个点的概率相同, [成长型]:X越大取到的概率越大, [抛物线型]:中间的X被取到的概率更大")
    
                create_x_mode_transfer = {
                    "均匀":"uniform",
                    "成长型":"grow",
                    "抛物线型":'parabola'
                }
    
                create_x_mode = create_x_mode_transfer[create_x_mode]
    
    
            with create_linear_intercept_col:
    
                create_linear_intercept_range = st.slider("线性函数截距范围", -25, 25, (-5, 5), key=f'{block_id}_create_linear_intercept_range', step=2)
    
                create_redun_slope = st.slider("X的冗余倾向 ⚠️", min_value=1.0, max_value=8.0, step=0.5, value=2.5,
                                               key=f"{block_id}_create_redun_slope",
                                               help="[⛓️‍存在依赖项]: 此项发挥作用需要 [开启冗余] 被设置为 [True],  \n"
                                                    "[⚠️激进的参数]: 此参数将显著影响函数复杂度,  \n"
                                                    "[冗余倾向] 越高,则越可能生成复杂的 [冗余项], 也越可能使用 [冗余特征] 参与构成 [y]  \n"
                                               )
    
                create_redun_seed = st.number_input("冗余随机种子", min_value=0, max_value=20000, step=1, value=73, key=f"{block_id}_create_redun_seed",
                                                   help="开启冗余后, 随机种子用于决定冗余的状态"
                                                   )


            create_xtox_bank = st.multiselect(label="可出现的互作函数:", options=["All"] + list(xtox_func_dict.keys()),
                                              default=["积函数", "绝对值和函数", "正弦和函数"],
                                              key=f"{block_id}_create_xtox_bank")

            if f"{block_id}_All" in create_xtox_bank:
                create_xtox_bank = list(xtox_func_dict.keys())

            create_thresh_range = st.slider("数据量限制在", 0, 10000, (300, 700), key=f"{block_id}_create_thresh_range", step=100)
    
            create_linear_limit_col, only_usedx_col = st.columns([2, 1])
    
            with create_linear_limit_col:
    
                create_linear_limit = st.checkbox("强制线性", key=f"{block_id}_create_linear_limit")
    
            with only_usedx_col:
    
                use_x_piked = st.checkbox("排除无关特征", key=f"{block_id}_use_x_piked", value=True)
    

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
    
            if f"{block_id}_eg_created" not in st.session_state:

                create_eg_kwargs = copy.deepcopy(create_kwargs)
                st.session_state[f"{block_id}_create_eg_kwargs"] = create_eg_kwargs
    
                x_eg, X_eg, y_exp_eg, x_picked_eg = gen_y_exp(**create_eg_kwargs, sample_num=eg_create_samples)
    
                st.session_state[f'{block_id}_x_eg'] = x_eg
                st.session_state[f"{block_id}_X_eg"] = X_eg
                st.session_state[f'{block_id}_y_exp_eg'] = y_exp_eg
                st.session_state[f'{block_id}_x_picked_eg'] = x_picked_eg
                eg_change=False
    
            else:
    
                eg_change = (st.session_state[f"{block_id}_create_eg_kwargs"] != create_kwargs)
                create_eg_kwargs = copy.deepcopy(create_kwargs)
                st.session_state[f"{block_id}_create_eg_kwargs"] = create_eg_kwargs
    
                if eg_change:

                    print('eg_changed')

                    x_eg, X_eg, y_exp_eg, x_picked_eg = gen_y_exp(**create_eg_kwargs, sample_num=eg_create_samples)
    
                    st.session_state[f'{block_id}_x_eg'] = x_eg
                    st.session_state[f"{block_id}_X_eg"] = X_eg
                    st.session_state[f'{block_id}_y_exp_eg'] = y_exp_eg
                    st.session_state[f'{block_id}_x_picked_eg'] = x_picked_eg
    
                else:
                    print('eg_not_changed')

                    x_eg = st.session_state[f'{block_id}_x_eg']
                    X_eg = st.session_state[f'{block_id}_X_eg']
                    y_exp_eg = st.session_state[f'{block_id}_y_exp_eg']
                    x_picked_eg = st.session_state[f'{block_id}_x_picked_eg']

    
            # 7-5-3 生成 eg ---------------------------------------------------------------------------------------------
    
            if use_x_piked:
    
                x_eg_use = copy.deepcopy(x_eg)[x_picked_eg] # 坑爹的 df.copy
    
            else:
                x_eg_use = copy.deepcopy(x_eg)
    
            np.random.seed(create_kwargs["x_seed"])
    
            x_eg_columns = x_eg.columns.tolist()
    
        formular_eg = 'y=' + '+'.join(list(X_eg.columns))
    
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
        st.markdown('')
        st.markdown('')
        st.markdown('')



        # 7-5-4 打印公式的latex格式 ---------------------------------------------------------------------------------------
    
        latex_expr = apply_colored_brackets(formular_eg)
        st.latex(latex_expr)

        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown('')
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        hr_second(dark_color="#244690", height=2.5, light_color="#26519d")
    
    # 独立与互作特征鉴别器(待升级)
    def count_unique_ai(expr: str, list_a: list) -> int:
        count = 0
        for item in list_a:
            if item in expr:
                count += 1
        return count
    
    
    # 读取颜色（默认浅色）


    with create_output_control_panel_:

        hr_second(dark_color="#244690", height=2.5, light_color="#26519d")

        formula_panel1_col1, formula_panel1_col2 = st.columns([.5, .5])

        with formula_panel1_col1:

            st.markdown("")
            st.markdown("")
            st.markdown(
                "<h3 style='text-align: center;'>数据推送</h3>",
                unsafe_allow_html=True
            )
            st.markdown("---")

        with formula_panel1_col2:

            st.markdown("")
            st.markdown("")
            st.markdown(
                "<h3 style='text-align: center;'>预览区 DPLS 控制</h3>",
                unsafe_allow_html=True
            )
            st.markdown("---")

    
        with formula_panel1_col2:
    
            # 使用 expander 创建一个可折叠/展开的区域
            DPLSR_param_dict_ = DPLSR_param_dict.copy()

            DPLS_kwargs = param_controller(
                param_list=['DPLSR'],
                para_descriptions={'DPLSR': method_descriptions.get('DPLSR')},
                param_controls={"DPLSR": DPLSR_param_dict_},
                desc='方法',
                a_copied_dict=f"{block_id}", expanded=True,
            )
            if not DPLS_kwargs['DPLSR']["distance_pattern"]:
                DPLS_kwargs['DPLSR']["distance_pattern"] = ['Euc']

        with formula_panel1_col1:

            create_exe_expander_2 = st.expander(f"**推送**", expanded=True)
    
            # 7-5-6-1 打印模拟样本的生成参数  -------------------------------------------------------------------------------
    
            with create_exe_expander_2:


                st.markdown("---")

                col_file_num, col_test_seed = st.columns([1, 1])
    
                with col_file_num:
                    test_files_num = st.number_input(
                        "生成类似函数的个数",
                        min_value=10,
                        max_value=10000,
                        value=20,
                        step=10
                        , key=f"{block_id}_test_files_num")
    
                    illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False], key=f"{block_id}_illegal_compatible_mode")

                with col_test_seed:
                    seed_value = st.number_input(
                        "类似函数抽样种子",
                        min_value=0,
                        max_value=20000,
                        value=42,
                        step=1
                        , key=f"{block_id}_seed_value")


                    print(f"in sym-{block_id}_seed_value", seed_value )

                    if st.session_state.get(f"{block_id}_gen_floor", False):

                        if st.session_state[f"{block_id}_gen_floor"] < test_files_num:

                            gen_floor = st.number_input(
                                "噪音梯度",
                                min_value=1,
                                max_value=test_files_num,
                                value=st.session_state[f"{block_id}_gen_floor"],
                                step=1
                                , key=f"{block_id}_gen_floor")

                        else:

                            gen_floor = st.number_input(
                                "噪音梯度",
                                min_value=1,
                                max_value=test_files_num,
                                value=test_files_num,
                                step=1
                                , key=f"{block_id}_gen_floor")
                    else:

                        gen_floor = st.number_input(
                            "噪音梯度",
                            min_value=1,
                            max_value=test_files_num,
                            value=1,
                            step=1
                            , key=f"{block_id}_gen_floor")

                st.markdown("---")

                create_noise = st.slider("y_obs 噪音强度域", min_value=0.0, max_value=5.0, step=0.05, value=(0.05, 0.5),
                                         key=f"{block_id}_create_noise",
                                         help="生成模拟样本后, 往[期望值y_exp]添加的噪音强度"
                                         )

                thresh_range = st.slider("样本数限制在", 0, 10000, (100, 1500), key=f"{block_id}_thresh_range", step=100)

                st.markdown("<div style='height:41px'></div>", unsafe_allow_html=True)

                output_create_button = st.button("**推送到项目**", use_container_width=True, key=f"{block_id}_output_create_button")


            # DPLS_attr_dict = DPLS().__dict__.copy()
            # DPLS_needed_param = ["cv", "max_iter", "R2", "cv_R2", "fit_R2", "p", "cv_p", "fit_p", "y_pred_R2", "square"]
            # DPLS_attr_dict = {k: v for k, v in DPLS_attr_dict.items() if k in DPLS_needed_param}
            #
            # with st.expander("**选择需要的 DPLS 属性**", expanded=True):
            #
            #     analys_direction = st.selectbox("属性方向", ["X - y", "y - X", "X - y&y - X"],
            #                                     key=f"{block_id}_analys_direction")
            #     analys_direction = direction_transform[analys_direction]
            #
            #     DPLS_attr_col = st.columns([1.32, 1])
            #
            #     DPLS_picked_attr = {}
            #
            #     for idx, key in enumerate(DPLS_attr_dict.keys()):
            #
            #         with DPLS_attr_col[idx % 2]:
            #             DPLS_picked_attr[key] = st.checkbox(f"{key}", value=False, key=f"{block_id}_DPLS_needed_attr_{key}")
            #
            # DPLS_picked_attr = {key:value for key, value in DPLS_picked_attr.items() if value}


    with create_preview_panel:
        st.markdown('')
        st.markdown('')
        st.subheader('图像')
        st.markdown('---')
        st.markdown("")
    
    
        plot_sep_0, plot_P_col, plot_sep_1,plot_addnoise_col, plot_sep_4, plot_y_exp_col, plot_sep_2, plot_y_obs_col, plot_sep_3, plot_DPLSR_col, plot_PrsR_col = st.columns(
            [.01, 1.3,.2,1.3,.2,.5,.2, .5, 1.2, 1.05, .9])
    
    
        with plot_P_col:
    
            enforce_P = st.slider("硬定位的 P: ", min_value=-1, max_value=DPLS_kwargs['DPLSR']['max_iter'] - 1,
                                  value=-1, key=f'{block_id}_enforce_P')
    
        with plot_addnoise_col:
    
            create_plot_noise = st.slider("y_obs 噪音强度", min_value=0.0, max_value=5.0, step=0.1, value=0.5,
                                     key=f"{block_id}_create_plot_noise",
                                     help="生成模拟样本后, 往[期望值y_exp]添加的噪音强度"
                                     )

            DPLS_kwargs['create_plot_noise'] = create_plot_noise

            y_obs_eg = y_exp_eg + np.random.normal(size=y_exp_eg.shape[0], loc=0,
                                                   scale=y_exp_eg.std() * create_plot_noise)
            y_df_eg = pd.DataFrame()
            y_df_eg["y_exp"] = y_exp_eg
            y_df_eg["y_obs"] = y_obs_eg
    
        with plot_y_exp_col:
            st.markdown("")
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            print_exp = st.checkbox('y_exp', value=True, key=f'{block_id}_print_exp')
    
    
        with plot_y_obs_col:
            st.markdown("")
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            print_obs = st.checkbox('y_obs', value=False, key=f'{block_id}_print_obs')
    
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown('')
        st.markdown("")
        st.markdown('')
    
        with st.spinner("正在拟合..."):

            # 检测 dpls 参数是否被改动过
            if f"{block_id}_eg_created" not in st.session_state:

                st.session_state[f"{block_id}_dpls_kwargs"] = DPLS_kwargs
                pred_obj = DPLS(**DPLS_kwargs["DPLSR"]).fit(x_eg_use.copy(), y_df_eg['y_obs'].copy(), **DPLS_kwargs["DPLSR"])
                st.session_state[f'{block_id}_pred_obj'] = pred_obj
                dpls_change = False

            else:

                dpls_change = (st.session_state.get(f"{block_id}_dpls_kwargs", None) != DPLS_kwargs)
                st.session_state[f"{block_id}_dpls_kwargs"] = DPLS_kwargs

                if dpls_change or eg_change:

                    pred_obj = DPLS(**DPLS_kwargs["DPLSR"]).fit(x_eg_use.copy(), y_df_eg['y_obs'].copy(),**DPLS_kwargs["DPLSR"])
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
            with plot_PrsR_col:

                if DPLS_kwargs["DPLSR"]["R_mode"] == 'single':
                    st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <h3 style='text-align: right; font-size: 28px; font-weight: bold;'>
                            Single Mode 
                        </h3>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <h3 style='text-align: right; font-size: 28px; font-weight: bold;'>
                            PrsR: 
                            <span style='color: #3580f5; font-size: 30px; font-weight: bold;'>
                                {calculate_corr(y_exp_eg, y_obs_eg)[0]:.2f}
                            </span>
                        </h3>
                        """,
                        unsafe_allow_html=True
                    )




        # 定义两列
        cols = st.columns(2)
        fig_dict = {}
        # 创建一个空图
    
        for idx, x_i in enumerate(x_picked_eg):
    
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
    
            if print_exp:
    
                x_eg_use_i['y_exp'] = y_df_eg['y_exp']
    
            if print_obs:
    
                x_eg_use_i['y_obs'] = y_df_eg['y_obs']
    
            x_eg_use_i['preds'] = y_pred  # 使用完整预测值
    
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
                yaxis_title=f'y',
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
                st.plotly_chart(fig, use_container_width=True, key=f"{block_id}_{x_i}create_plotly_chart")
    
            fig_dict[f"plots_{x_i}"] = fig


    create_check_file_sep, create_check_file_col = st.columns([0.001, 1])
    
    with create_check_file_col:

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

        created_data = {}
        total_0 = 0
        total_1 = 0
        total_file = 0

        if output_create_button and not(eg_change or dpls_change):

            print("run_created")

            with st.spinner('正在生模拟数据...'):

                create_times = test_files_num // (create_use_x_num + create_xtox_num)

                create_func_seeds = gen_seed(create_times, rand_seed=create_funcseed, gen_times=1)[0]
                create_x_seeds = gen_seed(create_times, rand_seed=create_xseed, gen_times=1)[0]

                noise_step = (create_noise[1] - create_noise[0]) / gen_floor
                create_noise_thresh = [(create_noise[0] + i * noise_step, create_noise[0] + (i + 1) * noise_step) for i
                                       in range(gen_floor)]

                for create_i in stqdm(range(create_times)):

                    np.random.seed(create_func_seeds[create_i])

                    create_kwargs["func_seed"] = create_func_seeds[create_i]
                    create_kwargs["x_seed"] = create_x_seeds[create_i]

                    eg_create_samples = np.random.randint(thresh_range[0], thresh_range[1])

                    # 7-4-1 参数选择完毕, 开始生成模拟数据 ----------------------------------------------------------------------

                    x_create, X_create, y_exp_create, x_picked_create = gen_y_exp(sample_num=eg_create_samples,
                                                                                  **create_kwargs)
                    np.random.seed(create_x_seeds[create_i])

                    create_noise_i = random.uniform(*create_noise_thresh[create_i % gen_floor])
                    y_obs_create = y_exp_create + np.random.normal(size=y_exp_create.shape[0], loc=0,
                                                                   scale=y_exp_create.std() * create_noise_i)
                    create_R_true = calculate_corr(y_exp_create, y_obs_create)[0]

                    create_func_name = 'y=' + '+'.join(list(X_create)) + f'[s-{create_func_seeds[create_i]}]'

                    if use_x_piked:

                        create_i_use = x_create.copy()[x_picked_create]

                    else:
                        create_i_use = x_create.copy()

                    create_i_use["y"] = y_obs_create
                    created_data_pair, created_pair_name, created_data_cause = return_cause_pair(create_i_use, prefix=f'[{create_func_name}]')

                    created_data[create_func_name] = {"files_pair": dict(zip(created_pair_name, created_data_pair)),
                                                      "files_cause": dict(zip(created_pair_name, created_data_cause)),
                                                      "R_true": dict(zip(created_pair_name, [create_R_true]*len(created_pair_name))),
                                                      "X_name":dict(zip(created_pair_name, x_picked_create)),
                                                      'description':'y=' + '+'.join(list(X_create)),}

                    count_0 = created_data_cause.count(0)
                    count_1 = created_data_cause.count(1)

                    total_0 += count_0
                    total_1 += count_1
                    total_file += len(created_data_pair)

            # 7-4-2 模拟数据存入全局变量: use_files_dict -----------------------------------------------------------------------

            use_files_dict = created_data.copy()
            st.session_state[f'{block_id}_use_files_dict'] = use_files_dict

            with create_exe_expander_2:

                st.success(f'已推送 {len(use_files_dict)} 个函数到项目')

            # 7-5-5 打印样本的 X 与 y -----------------------------------------------------------------------------------------

            # 7-5-6 打印模拟样本的生成参数 与 参数符合度的检测结果 ------------------------------------------------------------------

        # if output_create_button:
        #
        #     if DPLS_picked_attr:
        #
        #         all_files_dpls_values = []
        #
        #         for db_name, db_values in stqdm(use_files_dict.items()):
        #
        #             DPLS_obj_dict:dict = parallel_wrapper(cal_DPLS_obj, db_values["files_pair"], desc="cal_DPLS_obj", reason=0, result=1, thread=thread)
        #             db_values['files_dpls_obj'] = DPLS_obj_dict
        #             db_values['files_dpls_values'] = {key: {pick: dpls_obj.__dict__[pick] for pick in DPLS_picked_attr.keys()} for key, dpls_obj in db_values['files_dpls_obj'].items()}
        #
        #             db_files_dpls_values = pd.DataFrame.from_dict(db_values['files_dpls_values'], orient='index')
        #
        #             if "R2" in db_files_dpls_values.columns:
        #
        #                 R2_expanded = db_files_dpls_values['R2'].apply(lambda x: x[0])
        #                 db_files_dpls_values = pd.concat([db_files_dpls_values.drop(columns=["R2"]), R2_expanded], axis=1)
        #
        #             if "p" in db_files_dpls_values.columns:
        #
        #                 p_expanded = db_files_dpls_values['p'].apply(lambda x: x[0])
        #                 db_files_dpls_values = pd.concat([db_files_dpls_values.drop(columns=['p']), p_expanded], axis=1)
        #
        #             if 'y_pred_R2' in db_files_dpls_values.columns:
        #                 y_pred_R2_expanded = db_files_dpls_values['y_pred_R2'].apply(lambda x: x[0])
        #                 y_pred_R2_expanded =  y_pred_R2_expanded.apply(pd.Series)
        #                 y_pred_R2_expanded.columns = [f"r2_p{i}" for i in range(y_pred_R2_expanded.shape[1])]
        #                 db_files_dpls_values = pd.concat([db_files_dpls_values.drop(columns=["y_pred_R2"]), y_pred_R2_expanded], axis=1)
        #
        #             db_files_dpls_values["R_true"] = pd.Series(db_values["R_true"])
        #
        #             db_values['files_dpls_values'] = db_files_dpls_values
        #             all_files_dpls_values.append(db_files_dpls_values)
        #
        #         all_files_dpls_values = pd.concat(all_files_dpls_values, axis=0)
        #         st.dataframe(all_files_dpls_values)
        #
        #         csv = all_files_dpls_values.to_csv().encode('utf-8')
        #         # 添加下载按钮
        #         download_create_button = st.download_button(
        #             label="📥 下载",
        #             data=csv,
        #             file_name=f'create_{test_files_num}_data.csv',
        #             mime='text/csv',
        #             use_container_width=True, key=f"{block_id}_download_create_button"
        #         )
        #
        #     else:
        #         pass
        # if check_file_button or use_files_dict:
        #
        #     st.markdown("")
        #     st.markdown("")
        #     st.markdown("<h3 style='text-align: center;'>文件检视面板</h3>", unsafe_allow_html=True)
        #     st.markdown("")
        #     st.markdown("---")
        #
        #     expand_raw_now_files(use_files_dict, total_file=total_file, expand=True, print_pred=True, thread=thread)

    hr_second(dark_color="#244690", height=2.5, light_color="#26519d")

    file_param = {

        "illegal_compatible_mode": illegal_compatible_mode,
        'test_files_num': test_files_num,
        'seed_value': seed_value,
        'thresh_range': thresh_range,
        "create_redun": create_redun,
        "create_linear_limit": create_linear_limit,
        "gen_floor": gen_floor

    }

    return use_files_dict, file_param

    # with plot_download_col:
    #     with st.spinner(""):
    #
    #         zip_bytes = export_multiple_plots_to_zip(fig_dict)
    #
    #         st.markdown("")
    #         # 下载按钮
    #         st.download_button(
    #             label="📥",
    #             data=zip_bytes,
    #             file_name=f"all_plots_{formula_eg}.zip",
    #             mime="application/zip",
    #             use_container_width=True, key=f"{block_id}_plot_download_button"
    #         )

if __name__ == '__main__':

    simplify_data_module("test")
