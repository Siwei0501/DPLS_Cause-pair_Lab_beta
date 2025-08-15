import json
import os

import webbrowser

import statsmodels

from DPLS_GUI import *
from GUI_modules.create_func_module import create_func
from GUI_functions.DPLS_Checker import DPLS_Checker

st.markdown("""
<style>
/* 将整个内容区域整体往上移动 65px */
.block-container {
    position: relative;
    top: -65px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        min-width: 2300px !important;
        overflow-x: auto !important;
    }
    </style>
""", unsafe_allow_html=True)


# 0-2 设置路径 -----------------------------------------------------------------------------------------------------------


# 脚本路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, 'check-project_file')
now_version = "beta1"
st.set_page_config(page_title="DPLS Check Lab", layout="wide", page_icon='🟡')

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
    top: -20px;
    left: 35px;
    margin-bottom: 10px;
    margin-top: 25px;
}

/* 主标题样式 */
.custom-title {
    text-align: right;
    font-size: 2.6em !important;
    font-family: "Segoe UI Variable Text", "Roboto", "Helvetica Neue", sans-serif !important;
    margin-left: 50px !important;
    padding: 0;
}

/* Cause 渐变文字样式 */
.gradient-text {
    background: linear-gradient(30deg, #FFD93D, #FFc527, #FFa512, #FF4E50);
    background-size: 300% 300%;
    background-position: 0% 50%;
    animation: animated-gradient 9s ease infinite;
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
        DPLS&nbsp;<span class="gradient-text">Check&nbsp;</span>Laboratory
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
        top: -25px;
        left: 0px;
        height: 2px !important;  /* 可调粗细 */
        background: #333 !important;
        background-image: linear-gradient(to right, #FFD93D, #FFa512, #FF4E50) !important;
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
        background-image: linear-gradient(to right, #FFD93D, #FFa512, #FF4E50) !important;
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


project_name_col, project_go_col, project_sep1, project_rename_col, project_sep2, project_select_col = st.columns([.8,.2,.01, .35,.02, .7])

project_input = False
now_project_name = None

with project_go_col:

    run_button = st.button(r"**Project GO ▶**", use_container_width=True)

with project_sep2:
    st.markdown("""
        <style>
            .vertical-line_1 {
                width: 0.5px;
                height: 43px !important;
                margin: auto;
                margin-top: 0px;
            }

            @media (prefers-color-scheme: dark) {
                .vertical-line_1 {
                    background-color: #333333;
                }
            }

            @media (prefers-color-scheme: light) {
                .vertical-line_1 {
                    background-color: #d8d8d8;
                }
            }
        </style>

        <div class="vertical-line_1"></div>
    """, unsafe_allow_html=True)


if os.path.exists(project_dir):
    pass
else:
    os.makedirs(project_dir)

with project_select_col:

    project_files = [
        f for f in os.listdir(project_dir)
        if f.endswith('.json')
    ]

    project_files = sorted(
        project_files,
        key=lambda f: os.path.getmtime(os.path.join(project_dir, f)),
        reverse=True
    )

    select_project, load_project_, manage_project = st.columns([.42, .5, .08])

with select_project:
    selected_project = st.selectbox('选择运行项目文件', project_files, label_visibility='collapsed')

# 1-1-3 载入项目功能 ----------------------------------------------------------------------------------------------------

with load_project_:
    load_select_project_col, load_last_project_col = st.columns([.4, .8])

with load_select_project_col:
    load_select_1, load_select_2 = st.columns([.5, .5])


    with load_select_2:

        refresh_project_button = st.button('**↻**', use_container_width=True)

        if refresh_project_button:
            st.rerun()


with project_rename_col:
    project_rename_col_ = st.columns([6, 1])


with project_rename_col_[1]:

    save_project_button = st.button('💾', use_container_width=True)

    with load_select_1:

        load_project_button = st.button('✔', use_container_width=True)
        if load_project_button:

            if len(project_files) > 0:
                with open(os.path.join(project_dir, selected_project), 'r', encoding='utf-8') as f:
                    input_project = json.load(f)

                    project_input = True
                    st.session_state.update(input_project)

                    st.rerun()

with project_rename_col_[0]:


    now_project_name = st.text_input('本次项目文件名', key="now_project_name", placeholder='在此输入本项目名',
                                     help='输入文件名即可, 不需要加后缀如.json等', label_visibility='collapsed')

    now_project_name_print = now_project_name if now_project_name else "DPLS_Check Project"

with load_last_project_col:
    load_last_project = st.button('运行上次项目', use_container_width=True)

    if load_last_project:
        if os.path.exists(os.path.join(project_dir, 'last_run_project.json')):
            with open(os.path.join(project_dir, 'last_run_project.json'), "r", encoding="utf-8") as f:
                last_project = json.load(f)

            if "now_project_name" in st.session_state:
                del st.session_state["now_project_name"]

            st.session_state.update(last_project)
            project_input = True

            st.success("✔ 已成功载入上次运行项目")
            st.rerun()

        else:

            st.warning("⚠️ 没有找到上次运行项目")

with manage_project:
    manage_project_button = st.button("📁", use_container_width=True)

    if manage_project_button:
        webbrowser.open_new_tab(project_dir)

with project_name_col:

    project_name_col_1, project_name_col_2 = st.columns([.1, 1])

    with project_name_col_2:
        st.markdown(f"""
            <style>
            @media (prefers-color-scheme: light) {{
                .cause-title {{
                    color: #333333 !important;
                }}
            }}
            @media (prefers-color-scheme: dark) {{
                .cause-title {{
                    color: #ccc !important;
                }}
            }}
            </style>
            <div style='
                text-align: left;
                margin-top: 2px;
                margin-left: -20px;
            '>
                <span class='cause-title' style='
                    font-size: 24px;
                    font-weight: 650;
                    padding-bottom: 0px;
                    display: inline-block;
                '>{now_project_name_print}</span>
            </div>
        """, unsafe_allow_html=True)

        with project_name_col_1:

            st.button(f"**{now_project_name_print[:2]}**")



st.markdown("---")
st.sidebar.markdown('')

report_zone = st.container()

with report_zone:
    st.empty()

def error_to_report_zone(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            with report_zone:
                st.error(f"错误：{str(e)}")
    return wrapper

Thread = st.sidebar.slider("并行线程数: 轻量任务以及报错时设置值 = 1", 1, 16, 1, key='Thread')

st.sidebar.markdown('')
host_panel_file_detail,host_panel_sep, host_panel_funcs, = st.columns(
    [.5, .013, .8])


with host_panel_funcs:

    st.markdown("")
    st.markdown(
        "<h3 style='text-align: center;'>函数控制面板</h3>",
        unsafe_allow_html=True
    )
    hr_second(dark_color="#a06F26", height=2, light_color="#FFBB70")

with host_panel_file_detail:

    st.markdown("")
    st.markdown(
        "<h3 style='text-align: center;'>等价性图</h3>",
        unsafe_allow_html=True
    )
    hr_second(dark_color="#a06F26", height=2, light_color="#FFBB70")


with host_panel_funcs:
    st.markdown("")
    add_func_col, minus_func_col, check_sep, check_desc_col, check_notice_col= st.columns([.09,.09,.02, .5, .3])

    if st.session_state.get('use_files_dict', False):
        pass
    else:
        st.session_state['use_files_dict'] = {}

    use_files_dict = st.session_state['use_files_dict'].copy()

    if st.session_state.get('use_params_dict', False):
        pass
    else:
        st.session_state['use_params_dict'] = {}

    use_params_dict = st.session_state['use_params_dict'].copy()

    if st.session_state.get('plot_params_dict', False):
        pass
    else:
        st.session_state['plot_params_dict'] = {}

    plot_params_dict = st.session_state['plot_params_dict'].copy()


    if 'func_num' in st.session_state:
        func_num = st.session_state['func_num']
    else:
        func_num = 2

    with add_func_col:

        add_func = st.button("＋", key=f"add_func_button", use_container_width=True)

        if add_func and func_num<8:

            func_num += 1

    with minus_func_col:


        minus_func = st.button("－", key=f"minus_func_button", use_container_width=True)

        if minus_func and func_num>0:

            func_num -= 1

    with check_desc_col:

        plot_desc = st.text_input('图片标题', key='plot_desc', placeholder='在此输入图片标题', label_visibility='collapsed')
        if not plot_desc:

            plot_desc = 'No Title'

    with check_notice_col:

        plot_notice = st.text_input('图片备注', key='plot_notice', placeholder='图片备注',
                                  label_visibility='collapsed')

    plot_params_dict['func_num'] = func_num

    st.markdown(
        """
        <hr style="margin-top: 23px; margin-bottom: 10px;">
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    for f in range(func_num):

        f_files, f_params = create_func(block_id=f'func_{f}', x_num=f+1, text=f'func_{f+1}')

        use_files_dict[f] = f_files
        use_params_dict[f] = f_params

    st.session_state['func_num'] = func_num


with host_panel_sep:
    st.markdown(f"""
        <style>
            .vertical-line_2 {{
                width: 0.5px;
                height: {245+st.session_state['func_num']*266}px;
                margin: auto;
                margin-top: -5px;
            }}

            @media (prefers-color-scheme: dark) {{
                .vertical-line_2 {{
                    background-color: #333333;
                }}
            }}

            @media (prefers-color-scheme: light) {{
                .vertical-line_2 {{
                    background-color: #d8d8d8;
                }}
            }}
        </style>

        <div class="vertical-line_2"></div>
    """, unsafe_allow_html=True)


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


with host_panel_file_detail:

    check_sep_col, check_region_col, check_density_col, check_dpls_expander_col, check_plot_download_col = st.columns([.005, .25, .25, .5, .07])

    with check_region_col:

        check_region = st.slider('R 范围', min_value=0.0, max_value=1.0, step=0.01, value=(0.0, 1.0), key='check_region',
                                 help='此项可限制 [PrsR(y_exp, y_obs)] 的范围')

    with check_density_col:

        check_density = st.number_input('测试密度', min_value=30, max_value=500, step=10, value=300//func_num if func_num > 0 else 30,
                                        key='check_density', help='每个 [函数] 在 [R区间] 内点的密度')

    with check_dpls_expander_col:

        # 使用 expander 创建一个可折叠/展开的区域
        DPLSR_param_dict_ = DPLSR_param_dict.copy()
        st.markdown("<div style='height:21px'></div>", unsafe_allow_html=True)
        DPLS_kwargs = param_controller(
            param_list=['DPLSR'],
            para_descriptions={'DPLSR': method_descriptions.get('DPLSR')},
            param_controls={"DPLSR": DPLSR_param_dict_},
            desc='方法',
            a_copied_dict=f"dpls_check", expanded=False,
        )

        if not DPLS_kwargs['DPLSR']["distance_pattern"]:
            DPLS_kwargs['DPLSR']["distance_pattern"] = ['Euc']


    plot_params_dict['check_region'] = check_region
    plot_params_dict['check_density'] = check_density
    plot_params_dict['Thread']=Thread

    use_params_dict.update(DPLS_kwargs)

    preview_change = (use_params_dict != st.session_state['use_params_dict'] or plot_params_dict != st.session_state['plot_params_dict'])

    if preview_change and not st.session_state.get('checker_fig', False):

        with st.spinner("正在生成预览图像"):

            eg_N_DF = pd.DataFrame()

            for f, func_dict in use_files_dict.items():

                np.random.seed(None)
                eg_f = np.linspace(check_region[0], check_region[1], check_density)
                eg_f_noise_coef = np.random.uniform(-.03, .03, size=eg_f.shape)
                eg_f_noise = eg_f * ( 1 + eg_f_noise_coef)

                eg_N_DF[func_dict['func_name']] = eg_f_noise

            eg_R_DF = 1 - eg_N_DF

            # 生成同形状的噪声矩阵
            N_noise = np.random.uniform(-0.03, 0.03, size=eg_N_DF.shape)
            eg_N_DF = eg_N_DF * (1 + N_noise)

            eg_N_DF[eg_N_DF<0.0] = 0.0
            eg_N_DF[eg_N_DF>1.0] = 1.0
            eg_R_DF[eg_R_DF<0.0] = 0.0
            eg_R_DF[eg_R_DF>1.0] = 1.0

            eg_Checker_obj = DPLS_Checker()
            eg_fig = eg_Checker_obj.plot(x_df=eg_N_DF, y_df=eg_R_DF,details=True, desc=f"DPLS equity Preview [Not-Real result] {plot_desc}",
                                         x_label='DPLS_R^2', y_label='Noise (1 - R<y_exp, y_obs>^2)')

            st.pyplot(eg_fig)

            st.session_state['use_params_dict'] = use_params_dict
            st.session_state['plot_params_dict'] = plot_params_dict

    elif run_button:

        with st.spinner("正在生成等价性图"):

            checker_obj = DPLS_Checker(input_func_dict=use_files_dict, thread=Thread, dots_num=check_density,
                                       **DPLS_kwargs['DPLSR'])
            checker_fig = checker_obj.check_x_equity(

                region=(check_region[0], check_region[1]),
                desc=plot_desc,
                details=True,


            )
            st.session_state['checker_fig'] = fig_to_png_bytes(checker_fig)
            st.rerun()

    elif st.session_state.get('checker_fig', False):

        st.image(st.session_state.checker_fig)
        with check_plot_download_col:
            st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
            st.download_button(
                label="🡇",
                data=st.session_state.checker_fig,
                file_name=f"{plot_desc}.png",
                mime="image/png"
            )

    else:

        st.markdown(
            """
            <hr style="margin-top: -3px; margin-bottom: 10px;">
            """,
            unsafe_allow_html=True
        )



