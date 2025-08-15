
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from GUI_functions.muti_func_test import gen_y_exp
from DPLS_GUI import *
import copy

st.set_page_config(layout="wide")


# 7-2-2 与 本地文件模式共用的功能部分 --------------------------------------------------------------------------------

# 模拟样本的独立功能区, 单独于 host_panel_check_file 外 -------------------------------------------------------------------------

def create_func(block_id, x_num:int = 1, text:str="",  thread=1):

    file_param = {}

    create_control_panel, create_sep1, create_preview_panel = st.columns([.5, 0.026, .618])

    with create_preview_panel:

        preview_panel_sep, preview_panel_refresh = st.columns([1, .2])

        with preview_panel_sep:

            st.markdown(f'{text}', unsafe_allow_html=True)

        with preview_panel_refresh:

            refresh_button = st.button('⇆', key=f"{block_id}_preview_refresh_button", use_container_width=True)

            if refresh_button:

                st.session_state[f"{block_id}_formular_refresh"] = True

    with create_sep1:
        st.markdown("""
            <style>
                .vertical-line {
                    width: .5px;
                    height: 205px;
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


        # 7-5-1 定义内容区 -----------------------------------------------------------------------------------------------
        st.markdown("")
        formular_refresh_col, formular_sep_1, formular_title_add_x, add_x_col, minus_x_col, formular_sep_2, formular_title_add_xtox, add_xtox_col, minus_xtox_col = st.columns(
            [.5, .15, .35, .21, .21, .15, .35, .21, .21])

        st.markdown("---")

        create_func_control_panel_expander = st.expander("详细参数")

        # with formular_title_col:
        #     render_section_title("你的函数 be like :", underline=False)

        with formular_refresh_col:

            f_clear_button = st.button("Clear", use_container_width=True, key=f"{block_id}_clear_button")
            if f_clear_button:
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

            render_noline_title("独立项")

        st.session_state[f'{block_id}_add_x'] = False
        st.session_state[f'{block_id}_add_xtox'] = False
        st.session_state[f'{block_id}_minus_x'] = False
        st.session_state[f'{block_id}_minus_xtox'] = False

        with add_x_col:
            add_x = st.button("＋", key=f"{block_id}_add_x_button")
            if add_x:
                st.session_state[f"{block_id}_add_x"] = True

        with minus_x_col:
            minus_x = st.button("－", key=f"{block_id}_minus_x_button")
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
            render_noline_title("互作项")
        with add_xtox_col:
            add_xtox = st.button("＋", key=f"{block_id}_add_xtox_button")
            if add_xtox:
                st.session_state[f"{block_id}_add_xtox"] = True
        with minus_xtox_col:
            minus_xtox = st.button("－", key=f"{block_id}_minus_xtox_button")
            if minus_xtox:
                st.session_state[f"{block_id}_minus_xtox"] = True

    with create_func_control_panel_expander:

        create_help_dict = {

            "create_param_num": '这个参数决定 [特征池] 的大小,  参与 [y] 构成的那些自变量 [X] 将会在池中选择, 池中剩余的 [特征] 为[无关特征]',
            "create_use_x_num": "从 [特征池] 内取用的 [X] 的数量, 取用 [X] 数量 ≤ [特征池]大小",
            "create_x_num": "此参数定义了构成 [y] 的函数关系内有多少独立项, 如 [y = x_1 + x_1^2 + x_2 + (x_1×x_2)],  \n"
                            "此时 [独立项] = 3 (即除 (x_1×x_2) 项以外的项, 他们仅由一个 [x] 决定), 注意到这里出现了重复的 [x]: x_1,"
                            "因为此参数不关注 [x] 是否重复,  \n但受参数: [使用的X数量] 的约束, "
                            "因为被标记为需要使用的 [X] 必须在 [y] 的公式中至少出现一次, ",

            "create_redun": "如果此项为 [True], 则可以有 [x非独立生成], 以下提到的情况就是 [合理] 的:  \n"
                            " 设参数 [独立项数量] = 3 , 生成的函数关系为 [y=x_1 + 2×x_2 + 3×x_3]"
                            " 表面上 [3×x_3] 这一项仅由 [x_3]构成, 因此被视为 [独立项], "
                            "但 [x_3] 由关系: [x_3=x_1×x_2] 约束, 所以 [真实] 的函数关系为  \n"
                            " [y=x_1 + 2×x_2 + 3×(x_1×x_2))],实际上只有两个 [X]参与了函数关系  \n"
                            "[⚠️激进的参数]: 此项将会显著增加函数复杂度  \n",

            "create_xtox_num": "此参数定义了构成 [y] 的函数关系内有多少互作项, 互作列是指由[多个X]影响的列,"
                               " 只会由选择为[使用的X]构成,  \n如 [y = x_1 + x_1^2 + x_2 + (x_1×x_2)], "
                               "此时 [互作项] = 1",

            "create_xtox_level": "此项只限制了每个互作项的 [最高参与项数], 而 [不一定] 得到最高参与项,  \n"
                                 "如设置此项数为 4, 得到 [x_3=x_1×x_2×(x_2-x_1)], 此时 [x_3] 的项数 [=4],  \n"
                                 "但是更换随机种子得到 [x_3=x_1×x_2], 此时 [2<=4], 是合理的.  \n"
                                 "[总结]: 当此项设为 [n] 时, 将不会超过参与项超过 [n] 的 [互作项]",

            "create_funcseed": "输入x数量后, 每个x被分到一个f构成f(x), f不一定一样, 随机种子用于决定这些 f",

            "create_x_bank": "'正弦函数': f'sin(πx)',  \n"
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

        create_main_sets_col_1, create_main_sets_col_2 = st.columns([.5, .5])

        # 7-2-3-1 无关功能 -------------------------------------------------------------------------------------------

        with create_main_sets_col_1:

            unrelated_expander = st.expander("**无关**", expanded=True)

            with unrelated_expander:

                create_use_x_col, create_x_col = st.columns([1, 1])

                with create_x_col:

                    create_interact_num = st.number_input("无关的X数量", min_value=0, max_value=20, step=1, value=0,
                                                          key=f"{block_id}_create_param_num",
                                                          help=create_help_dict.get("create_param_num", "无描述")
                                                          )

                with create_use_x_col:

                    if f"{block_id}_create_use_x_num" in st.session_state:

                        if st.session_state.get(f"{block_id}_add_x", False) or st.session_state.get(
                                f"{block_id}_add_xtox", False):

                            create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                               max_value=20,
                                                               step=1, value=st.session_state[
                                                                                 f"{block_id}_create_use_x_num"] + 1,
                                                               key=f"{block_id}_create_use_x_num",
                                                               help=create_help_dict.get("create_use_x_num", "无描述"),
                                                               )

                        elif st.session_state.get(f"{block_id}_minus_x", False) or st.session_state.get(
                                f"{block_id}_minus_xtox", False):

                            if st.session_state[f"{block_id}_create_use_x_num"] > 1:
                                create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                                   max_value=20,
                                                                   step=1, value=st.session_state[
                                                                                     f"{block_id}_create_use_x_num"] - 1,
                                                                   key=f"{block_id}_create_use_x_num",
                                                                   help=create_help_dict.get("create_use_x_num",
                                                                                             "无描述"),
                                                                   )
                            else:

                                create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                                   max_value=20,
                                                                   step=1, value=1, key=f"{block_id}_create_use_x_num",
                                                                   help=create_help_dict.get(f"create_use_x_num",
                                                                                             "无描述"),
                                                                   )

                        else:

                            create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                               max_value=20,
                                                               step=1,
                                                               value=st.session_state[f"{block_id}_create_use_x_num"],
                                                               key=f"{block_id}_create_use_x_num",
                                                               help=create_help_dict.get("create_use_x_num", "无描述"),
                                                               )


                    else:

                        create_use_x_num = st.number_input("使用的X数量", min_value=1,
                                                           max_value=20,
                                                           step=1, value=x_num, key=f"{block_id}_create_use_x_num",
                                                           help=create_help_dict.get("create_use_x_num", "无描述"),
                                                           )

                    create_param_num = create_interact_num + create_use_x_num

            # 7-2-3-2 冗余功能 -------------------------------------------------------------------------------------------

            redun_expander = st.expander("**冗余**", expanded=True)

            with redun_expander:

                create_x_num_col, create_redun_col = st.columns([1, 1])

                with create_x_num_col:

                    if st.session_state.get(f"{block_id}_create_xtox_num", 0) == 0:

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

                        elif st.session_state.get(f"{block_id}_minus_x", False) and st.session_state[
                            f"{block_id}_create_x_num"] > create_x_limit:

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
                                                       value=x_num,
                                                       key=f"{block_id}_create_x_num",
                                                       help=create_help_dict["create_x_num"],
                                                       )

                with create_redun_col:

                    create_redun = st.selectbox("**开启冗余** ⚠️", [False, True], key=f"{block_id}_create_redun",
                                                help=create_help_dict.get("create_redun", "无描述"), )

        # 7-2-3-3互作功能 --------------------------------------------------------------------------------------------

        with create_main_sets_col_2:

            interaction_expander = st.expander("**互作**", expanded=True)

            with interaction_expander:

                create_xtox_num_col, create_xtox_level_col = st.columns([1, 1])

                with create_xtox_num_col:

                    if st.session_state.get(f"{block_id}_create_x_num", 2) == 0:

                        create_xtox_limit = 1

                    else:
                        create_xtox_limit = 0

                    if f"{block_id}_create_xtox_num" in st.session_state:

                        if st.session_state.get(f"{block_id}_add_xtox", False):

                            create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10,
                                                              step=1,
                                                              value=st.session_state[f"{block_id}_create_xtox_num"] + 1,
                                                              key=f"{block_id}_create_xtox_num",
                                                              help=create_help_dict.get("create_xtox_num", "无描述"),
                                                              )

                        elif st.session_state.get(f"{block_id}_minus_xtox", False) and st.session_state[
                            f"{block_id}_create_xtox_num"] > create_xtox_limit:

                            create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10,
                                                              step=1,
                                                              value=st.session_state[f"{block_id}_create_xtox_num"] - 1,
                                                              key=f"{block_id}_create_xtox_num",
                                                              help=create_help_dict.get("create_xtox_num", "无描述"),
                                                              )

                        else:

                            create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10,
                                                              step=1,

                                                              value=st.session_state[f"{block_id}_create_xtox_num"],
                                                              key=f"{block_id}_create_xtox_num",
                                                              help=create_help_dict.get("create_xtox_num", "无描述"),
                                                              )

                    else:
                        create_xtox_num = st.number_input("互作项数量", min_value=create_xtox_limit, max_value=10,
                                                          step=0, value=0,
                                                          key=f"{block_id}_create_xtox_num",
                                                          help=create_help_dict.get("create_xtox_num", "无描述"),
                                                          )

                with create_xtox_level_col:

                    create_xtox_level = st.number_input("互作参与项", min_value=2, max_value=5, step=1, value=2,
                                                        key=f"{block_id}_create_xtox_level",
                                                        help=create_help_dict.get("create_xtox_level", "无描述"),
                                                        )

            # 7-2-3-4 随机种子 -------------------------------------------------------------------------------------------

            creation_seed_expander = st.expander("**随机种子**", expanded=True)

            with creation_seed_expander:

                create_funcseed_col, create_xseed_col = st.columns([1, 1])

                with create_funcseed_col:

                    if st.session_state.get(f"{block_id}_formular_refresh", False) and st.session_state.get(f"{block_id}_create_funcseed", False):

                        print('run_refresh')

                        create_funcseed = st.number_input("f(x)随机种子", min_value=0, max_value=20000, step=1,
                                                          value=st.session_state[f"{block_id}_create_funcseed"] + 1,
                                                          key=f"{block_id}_create_funcseed",
                                                          help=create_help_dict.get("create_funcseed", "无描述"),
                                                          )

                        st.session_state[f"{block_id}_formular_refresh"] = False


                    elif st.session_state.get(f"{block_id}_create_funcseed", False):

                        create_funcseed = st.number_input("f(x)随机种子", min_value=0, max_value=20000, step=1,
                                                          value=st.session_state[f"{block_id}_create_funcseed"], key=f"{block_id}_create_funcseed",
                                                          help=create_help_dict.get("create_funcseed", "无描述"),
                                                          )

                    else:

                        create_funcseed = st.number_input("f(x)随机种子", min_value=0, max_value=20000, step=1,
                                                          value=74, key=f"{block_id}_create_funcseed",
                                                          help=create_help_dict.get("create_funcseed", "无描述"),
                                                          )



                with create_xseed_col:

                    create_xseed = st.number_input("x的随机种子", min_value=0, max_value=20000, step=1, value=409,
                                                   key=f"{block_id}_create_xseed",
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
                create_linear_coef_range = st.slider("线性函数系数范围", -10, 10, (-3, 3),
                                                     key=f"{block_id}_create_linear_coef_range", step=1)

                create_redun_count = st.slider("X的冗余最大层 ⚠️", min_value=1, max_value=8, step=1, value=5,
                                               key=f"{block_id}_create_redun_count",
                                               help=
                                                    "[互作层级]: 指原变量外嵌套的 f 层级, 如 x_3 = sin(x_2), 此时 [嵌套层级] = 1  \n"
                                                    " x_3 = cos(sin(x_2))^2, 此时 [嵌套层级] = 3  \n"
                                                    "[互作层级] 越高,则越可能生成复杂的 [冗余项]  \n"
                                                    "[⛓️‍存在依赖项]: 此项发挥作用需要 [开启冗余] 被设置为 [True],  \n"
                                                    "[⚠️激进的参数]: 此参数将显著影响函数复杂度,  \n"
                                               )

                create_x_mode = st.selectbox("X取值模式", ["均匀", "成长型", "抛物线型"],
                                             key=f"{block_id}_create_x_mode",
                                             help="[均匀]模式时,X在定义域内取到每个点的概率相同, [成长型]:X越大取到的概率越大, [抛物线型]:中间的X被取到的概率更大")

                create_x_mode_transfer = {
                    "均匀": "uniform",
                    "成长型": "grow",
                    "抛物线型": 'parabola'
                }

                create_x_mode = create_x_mode_transfer[create_x_mode]

            with create_linear_intercept_col:
                create_linear_intercept_range = st.slider("线性函数截距范围", -25, 25, (-5, 5),
                                                          key=f'{block_id}_create_linear_intercept_range', step=2)

                create_redun_slope = st.slider("X的冗余倾向 ⚠️", min_value=1.0, max_value=8.0, step=0.5, value=2.5,
                                               key=f"{block_id}_create_redun_slope",
                                               help="[冗余倾向] 越高,则越可能生成复杂的 [冗余项], 也越可能使用 [冗余特征] 参与构成 [y]  \n"
                                                    "[⛓️‍存在依赖项]: 此项发挥作用需要 [开启冗余] 被设置为 [True],  \n"
                                                    "[⚠️激进的参数]: 此参数将显著影响函数复杂度,  \n"
                                               )

                create_redun_seed = st.number_input("冗余随机种子", min_value=0, max_value=20000, step=1, value=73,
                                                    key=f"{block_id}_create_redun_seed",
                                                    help="开启冗余后, 随机种子用于决定冗余的状态"
                                                    )

            create_xtox_bank = st.multiselect(label="可出现的互作函数:", options=["All"] + list(xtox_func_dict.keys()),
                                              default=["积函数", "绝对值和函数", "正弦和函数"],
                                              key=f"{block_id}_create_xtox_bank")

            if f"{block_id}_All" in create_xtox_bank:
                create_xtox_bank = list(xtox_func_dict.keys())

            create_thresh_range = st.slider("数据量限制在", 0, 10000, (300, 700), key=f"{block_id}_create_thresh_range",
                                            step=100)

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
                eg_change = False

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

                x_eg_use = copy.deepcopy(x_eg)[x_picked_eg]  # 坑爹的 df.copy

            else:
                x_eg_use = copy.deepcopy(x_eg)

            np.random.seed(create_kwargs["x_seed"])

        formular_eg = 'y=' + '+'.join(list(X_eg.columns))
        st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)

        return_dict = {'x':x_eg_use, 'y_exp':y_exp_eg, 'x_picked':x_picked_eg, 'X':X_eg ,'func_name': formular_eg}

        # 7-5-4 打印公式的latex格式 ---------------------------------------------------------------------------------------

        latex_expr = apply_colored_brackets(formular_eg)
        st.latex(latex_expr)


    hr_second(dark_color="#a06F26", height=2, light_color="#FFBB70")

    file_param.update(create_kwargs)

    return return_dict, file_param













if __name__ == '__main__':

    create_func("func")