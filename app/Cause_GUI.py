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

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# 导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cause_pair_functions.casual_pair_tester import process, algorithms, return_values_DF
from cause_pair_functions.muti_func_test import gen_y_exp
from simplify_data_module import simplify_data_module
from custom_html_module import *

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

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        min-width: 2400px !important;
        overflow-x: auto !important;
    }
    </style>
""", unsafe_allow_html=True)


# 0-2 设置路径 -----------------------------------------------------------------------------------------------------------


# 脚本路径
script_dir = os.path.dirname(os.path.abspath(__file__))
#本地数据路径
local_data_dir = os.path.join(script_dir, 'Cause_DBs')
#项目路径
project_dir = os.path.join(script_dir, 'config_file')
#版本号
now_version = "beta4"

# 全局功能函数定义区 -------------------------------------------------------------------------------------------------------




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
        top: -25px;
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

project_name_col, project_go_col, project_sep1, project_rename_col, project_sep2, project_select_col = st.columns([.8,.2,.01, .35,.02, .7])

project_input = False


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

    with load_select_1:

        load_project_button = st.button('✔', use_container_width=True)
        if load_project_button:

            if len(project_files) > 0:
                with open(os.path.join(project_dir, selected_project), 'r', encoding='utf-8') as f:
                    input_project = json.load(f)

                    print("classify_shuffle_seed", input_project["classify_shuffle_seed"])

                    project_input = True
                    st.session_state.update(input_project)
                    st.toast(f'已读取项目: {selected_project[:-5]}')

                    print("classify_shuffle_seed", st.session_state["classify_shuffle_seed"])

                    st.rerun()

    with load_select_2:

        refresh_project_button = st.button('**↻**', use_container_width=True)

        if refresh_project_button:
            st.rerun()

with project_rename_col:
    project_rename_col_ = st.columns([6, 1])

    with project_rename_col_[0]:
        now_project_name = st.text_input('本次项目文件名', key="now_project_name", placeholder='在此输入本项目名',
                                         help='输入文件名即可, 不需要加后缀如.json等', label_visibility='collapsed')
    with project_rename_col_[1]:
        save_project_button = st.button('💾', use_container_width=True)

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
    st.markdown("""
        <style>
        @media (prefers-color-scheme: light) {
            .cause-title {
                color: #333333 !important;
            }
        }
        @media (prefers-color-scheme: dark) {
            .cause-title {
                color: #ccc !important;
            }
        }
        </style>
        <div style='
            text-align: left;
            margin-top: -1px;
            margin-left: 8px;
        '>
            <span class='cause-title' style='
                font-size: 24px;
                font-weight: 800;
                padding-bottom: 0px;
                display: inline-block;
            '>Cause-pair Project</span>
        </div>
    """, unsafe_allow_html=True)
st.markdown("---")


use_data_options = ["模拟样本", "本地样本", "上传样本"]
use_data_type = st.sidebar.selectbox("**选择分析的数据种类**", options=use_data_options, key="use_data_type")
st.sidebar.markdown('---')
st.sidebar.markdown('')
Thread = st.sidebar.slider("并行线程数: 轻量任务以及报错时设置值 = 1", 1, 16, 1, key='Thread')
st.sidebar.markdown('')



host_panel_files, host_panel_sep, host_panel_preprocess, host_panel_sep1, host_panel_file_detial = st.columns(
    [.24, .013, .24, .03, .8])


# 脚本路径
script_dir = os.path.dirname(os.path.abspath(__file__))
#本地数据路径
local_data_dir = os.path.join(script_dir, 'Cause_DBs')


# 读取文件描述
def read_txt_or_default(filepath):
    description_path = os.path.join(filepath, 'description.txt')

    try:
        with open(description_path, 'r', encoding='utf-8') as f:
            return f.read()
    except (FileNotFoundError, PermissionError) as e:
        return "No description"


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

    description_path = os.path.join(local_data_dir, dataset_name, 'description.txt')

    try:
        with open(description_path, 'r', encoding='utf-8') as f:
            description =  f.read()
    except (FileNotFoundError, PermissionError) as e:
        description =  "No description"

    return pair_presented_data, col_names, cause_y, description


st.set_page_config(layout="wide")

use_data_options = ["模拟样本", "本地样本", "上传样本"]

host_next = st.container()

with host_panel_sep1:
    st.markdown("""
        <style>
            .vertical-line_2 {
                width: 0.5px;
                height: 978px;
                margin: auto;
                margin-top: -5px;
            }

            @media (prefers-color-scheme: dark) {
                .vertical-line_2 {
                    background-color: #333333;
                }
            }

            @media (prefers-color-scheme: light) {
                .vertical-line_2 {
                    background-color: #d8d8d8;
                }
            }
        </style>

        <div class="vertical-line_2"></div>
    """, unsafe_allow_html=True)


if st.session_state.get(f"use_files_dict", False):

    use_files_dict = st.session_state.get(f"use_files_dict")
    print('get use_files_dict')

else:

    use_files_dict = {}
    print('not get use_files_dict')


if st.session_state.get(f"processed_files_dict", False):

    processed_files_dict = st.session_state.get(f"processed_files_dict")
    print('get processed_files_dict')

else:

    processed_files_dict = {}
    print('processed_files_dict')


# 3-3 选择预处理
with host_panel_preprocess:
    st.markdown("")
    st.markdown("")
    st.markdown(
        "<h3 style='text-align: center;'>预处理</h3>",
        unsafe_allow_html=True
    )
    hr_second(dark_color="#244690", height=2, light_color="#26519d")
    host_preprocess_expander = st.expander('**Preprocess**', expanded=True)

    with host_preprocess_expander:
        preprocess_selection = st.multiselect(
            label="预处理流程（可多选）", key="preprocess_selection", label_visibility='collapsed',
            placeholder='预处理流程 (可多选)',
            options=list(process.keys())
        )

        st.markdown("---")

        if preprocess_selection:
            preprocess_kwargs = param_controller(
                param_list=preprocess_selection,
                para_descriptions=preprocess_descriptions,
                param_controls=preprocess_param_controls, expanded=True,
                desc='预处理'
            )

def do_process_one(files_pair, preprocess_selection_, reverse=False):

    # 默认第一列为原因
    if reverse != 1:
        reason = 0
        result = 1
    else:
        reason = 1
        result = 0

    print(files_pair.copy)
    pairs_processed = files_pair.copy()

    for p, preprocess in enumerate(preprocess_selection_):

        if preprocess in process:

            pairs_processed = parallel_wrapper(func=process[preprocess], file_value_dict=pairs_processed,
                                               desc=preprocess, thread=Thread, reason=reason, result=result,
                                               seed=seed_value, **preprocess_kwargs.get(preprocess, {}),
                                               )

    return pairs_processed


def do_process(raw, preprocess_selection_, reverse=False) -> dict:

    raw_copy = raw.copy()

    if reverse:
        prefix = "B->A"

    else:
        prefix = "A->B"


    raw_pairs_num = len(raw['files_pair'])

    for db_keys, db_key_values in raw_copy.items():

        if len(db_key_values) == raw_pairs_num and isinstance(db_key_values, dict):

            db_key_values_ = {f"{prefix}_{k}": v for k, v in db_key_values.items()}
            raw_copy[db_keys] = db_key_values_

        else:

            pass


    pairs_processed = raw_copy['files_pair']
    pairs_processed = do_process_one(pairs_processed, preprocess_selection_, reverse=reverse)

    raw_copy['files_pair'] = pairs_processed

    return raw_copy


total_0 = 0
total_1 = 0
total_file = 0

# 7-1 分析本地数据-----------------------------------------------------------------------------------------------------
raw_files_dict = {}

if use_data_type == use_data_options[1]:

    # 3-1-1 读取本地数据 ----------------------------------------------------------------------------------------------

    local_datasets = [name for name in os.listdir(local_data_dir) if os.path.isdir(os.path.join(local_data_dir, name))]

    local_datasets = sorted(local_datasets)

    # 读取本地数据的描述文件
    database_descriptions = {dataset_dir: read_txt_or_default(os.path.join(local_data_dir, dataset_dir)) for
                             dataset_dir in local_datasets}

    # 7-1-2 定义分析本地数据的控制面板 -----------------------------------------------------------------------------------

    with host_panel_files:

        st.markdown("")
        st.markdown("")
        st.markdown(
            "<h3 style='text-align: center;'>选择文件</h3>",
            unsafe_allow_html=True
        )
        hr_second(dark_color="#244690", height=2, light_color="#26519d")

    with host_panel_files:

        local_file_panel_expander = st.expander("**Local_files**", expanded=True)

    with local_file_panel_expander:

        # 选择要分析的数据集
        database_selections = st.multiselect(label='分析本地数据集 (可多选)',
                                             key="database_selections", placeholder="本地数据集 (可多选)",
                                             options=list(local_datasets),
                                             default=list(local_datasets)[0], label_visibility='collapsed')

        st.markdown("---")

        data_sort_mode = st.selectbox(label='文件排序',
                                      key="data_sort_mode",
                                      options=["按首字母", "按大小"],
                                      )

        illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False], key="illegal_compatible_mode")

        direction_panel = st.columns([1, 1])

        # 文件方向
        with direction_panel[0]:
            file_direction = st.selectbox("样本方向", ["AB&BA", "AB", "BA", ], key="file_direction")

        # 分析方向
        with direction_panel[1]:
            preprocess_direction = st.selectbox("预处理方向", ["AB&BA", "AB", "BA"], key="preprocess_direction")

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

        check_file_button = st.button("**读取文件**", use_container_width=True)
        check_file_with_DPLS_button = st.button("**读取并预测 DPLS**", use_container_width=True)

    # 7-1-4 收集参数 -------------------------------------------------------------------------------------------------

    file_param = {
        "正在使用": f"{use_data_type}",
        "非法样本兼容": illegal_compatible_mode,
        '样本方向': file_direction,
        '预处理方向': preprocess_direction,
        '文件抽样数': test_files_num,
        '随机种子': seed_value,
        '样本量': thresh_range,

    }

    db_select_kwargs = {'relation': file_direction, 'threshold': list(thresh_range), 'seed': seed_value, 'test_SAMPLE': test_files_num}

elif use_data_type == use_data_options[0]:

    with host_next:

        raw_files_dict, file_param = simplify_data_module('Cause_GUI', thread=Thread)


    # 7-2-2 与 本地文件模式共用的功能部分 ----------------------------------------------------------------------------
    with host_panel_files:

        st.markdown("")
        st.markdown("")
        st.markdown(
            "<h3 style='text-align: center;'>选择文件</h3>",
            unsafe_allow_html=True
        )
        hr_second(dark_color="#244690", height=2, light_color="#26519d")
        create_file_panel_expander = st.expander("**Created_files**", expanded=True)


    with (create_file_panel_expander):

        if raw_files_dict:
            create_db_default = ["All"]

        else:

            create_db_default = []

        print("create_db_default", create_db_default)

        # 选择要分析的数据集
        database_selections = st.multiselect(label='模拟数据集 (可多选)',
                                             key="database_selections", placeholder="本地数据集 (可多选)",
                                             options=create_db_default + list(raw_files_dict.keys()),
                                             default=create_db_default, label_visibility='collapsed')

        if f"All" in database_selections:
            database_selections = list(raw_files_dict.keys())

        st.markdown("---")

        illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False], key="illegal_compatible_mode")

        col_file_num, col_test_seed = st.columns([1, 1])

        with col_file_num:
            test_files_num = st.number_input(
                "文件抽样数",
                min_value=10,
                max_value=file_param.get('test_files_num', 20),
                value=file_param.get('test_files_num', 20),
                step=10
                , key="test_files_num")

        with col_test_seed:
            seed_value = st.number_input(
                "文件抽样种子",
                min_value=0,
                max_value=20000,
                value=file_param.get('seed_value', 42),
                step=1
                , key="seed_value")

        check_file_button = st.button("**读取文件**", use_container_width=True)
        check_file_with_DPLS_button = st.button("**读取并预测 DPLS**", use_container_width=True)

    if not create_db_default:
        with host_panel_files:
            st.markdown("")
            st.markdown("")
            st.markdown(""" <div style="text-align: center; font-weight:300; font-size: 18px;"> ⚠️ 没有检测到数据, 请生成模拟数据后推送至项目 </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

    db_select_kwargs = {'relation': "AB", 'threshold': file_param['thresh_range'], 'seed': seed_value, 'test_SAMPLE': test_files_num}


    # 7-3 定义[上传]数据参数控制面板 --------------------------------------------------------------------------------------------

elif use_data_type == use_data_options[2]:

    with host_panel_files:

        st.markdown("")
        st.markdown("")
        st.markdown(
            "<h3 style='text-align: center;'>选择文件</h3>",
            unsafe_allow_html=True
        )
        hr_second(dark_color="#244690", height=2, light_color="#26519d")
        upload_file_panel_expander = st.expander("**Uploaded_files**", expanded=True)

    with upload_file_panel_expander:

        uploaded_files_dict = {}

        uploaded_files = st.file_uploader(
            "支持多文件上传, 默认最后一列为结果",
            type=["csv", "txt", "xlsx"],
            accept_multiple_files=True,
            help='目前 beta 版仅支持形状为两列的数据'
        )

        file_sep = st.text_input(r"文件分隔符 (如果有)", placeholder="输入分隔符, 不需要加引号",
                                         help="常见的分隔符有: <;>, </t>, </s>, 输入不用带<>", label_visibility='collapsed')
        st.markdown("")
        has_header = st.checkbox("文件包含列名 (将自动去除)", value=False)


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


    upload_files_cause = ['Unknown'] * len(use_files_dict.keys())
    raw_files_dict["Upload_files"] = {"files_pair": uploaded_files_dict, "files_cause": dict(zip(use_files_dict.keys(), upload_files_cause))}
    total_file = len(use_files_dict)


    # 7-3-2 定义[上传]数据参数控制面板 ----------------------------------------------------------------------------------


    with upload_file_panel_expander:

        st.markdown("---")

        illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False], key="illegal_compatible_mode")

        col_file_num, col_test_seed = st.columns([1, 1])

        with col_file_num:
            test_files_num = st.number_input(
                "文件抽样数",
                min_value=0,
                max_value=len(uploaded_files_dict),
                value=len(uploaded_files_dict),
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

        file_param = {
            "正在使用": f"{use_data_type}",
            "非法样本兼容": illegal_compatible_mode,
        }

        check_file_button = st.button("**读取**", use_container_width=True)
        check_file_with_DPLS_button = st.button("**读取并预测 DPLS**", use_container_width=True)

    db_select_kwargs = {'relation': "AB", 'threshold': thresh_range, 'seed': seed_value,
                        'test_SAMPLE': test_files_num}

else:

    raise AttributeError(f'Do not understand parameter {use_data_type}')

if use_data_type != use_data_options[0]:
    hr_second()

method_panel_col, method_panel_sep, method_panel_file_detial= st.columns(
    [.496,.03, .8])

hr_second()

classify_panel_col, classify_panel_sep, classify_panel_file_detial= st.columns(
    [.496,.03, .8])

with classify_panel_sep:
    st.markdown("""
        <style>
            .vertical-line_4 {
                width: 0.5px;
                height: 950px;
                margin: auto;
                margin-top: -5px;
            }

            @media (prefers-color-scheme: dark) {
                .vertical-line_4 {
                    background-color: #333333;
                }
            }

            @media (prefers-color-scheme: light) {
                .vertical-line_4 {
                    background-color: #d8d8d8;
                }
            }
        </style>

        <div class="vertical-line_4"></div>
    """, unsafe_allow_html=True)

with classify_panel_col:

    st.markdown("")
    st.markdown("")
    st.markdown("<h3 style='text-align: center;'>分类</h3>", unsafe_allow_html=True)
    st.markdown("---")

    classify_panel_expander =  st.expander("classifys", expanded=True)


with classify_panel_expander:

    # 5-4 选择分类器
    classify_selection = st.multiselect("分类方法 (可多选)", ['All'] + list(classify_dict.keys()),
                                                key="classify_selection")
    if "All" in classify_selection:
        classify_selection = list(classify_dict.keys())

    # 5-5 侧边栏分类器输入区
    classify_shuffle_1, classify_shuffle_2 = st.columns([1, 1])
    with classify_shuffle_1:
        cv_mode = st.selectbox('训练集选取模式', options=['uniform', 'layers'], key="cv_mode")

    with classify_shuffle_2:
        classify_shuffle_seed = st.number_input('样本混淆种子', value=42, key="classify_shuffle_seed")

    st.markdown('')
    classify_cv_col, classify_cv_mode_col = st.columns([3, 1])

    with classify_cv_col:
        classify_cv = st.slider('多重检验折数', min_value=1, max_value=10, value=5, step=1, key="classify_cv")

    with classify_cv_mode_col:
        st.markdown("<div style='height:37px;'></div>", unsafe_allow_html=True)
        classify_shuffle_confirm = st.checkbox('分类前混淆样本顺序', value=False, key="classify_shuffle_confirm")

    classify_output_button = st.button('**输出**', key='classify_output_button', use_container_width=True)

    hr_second()

    # 5-6 分类器打印内容
    if classify_selection:
        classify_print = {r + 1: classify_ for r, classify_ in enumerate(classify_selection)}
    else:
        classify_print = {}

    if classify_selection:

            st.markdown('')
            classify_kwargs = param_controller(
                param_list=classify_selection,
                para_descriptions=classify_description,
                param_controls=classify_param_control,
                desc='分类',cols=2
            )



# 分类方法执行器
def do_classify(classify_name, file_values: pd.DataFrame, **kwargs):
    # global title_bar_progress
    # global total_mission
    # global now_mission

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
            'min_samples_split': [2, 10],
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

    train_list, test_list = spliter(X.shape[0], cv=classify_cv, mode=cv_mode, random_before=classify_shuffle_confirm,
                                    shuffle_seed=classify_shuffle_seed, )

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

        # now_mission += 1
        # title_bar_progress.progress(now_mission/total_mission)

        return accuracy_of_preds

    except Exception as e:
        # now_mission += 1
        st.error(e)


with classify_panel_file_detial:

    st.markdown("")
    st.markdown("")
    st.markdown("<h3 style='text-align: center;'>方法值的分类结果</h3>", unsafe_allow_html=True)
    st.markdown("---")



with method_panel_sep:
    st.markdown("""
        <style>
            .vertical-line_3 {
                width: 0.5px;
                height: 1388px;
                margin: auto;
                margin-top: -5px;
            }

            @media (prefers-color-scheme: dark) {
                .vertical-line_3 {
                    background-color: #333333;
                }
            }

            @media (prefers-color-scheme: light) {
                .vertical-line_3 {
                    background-color: #d8d8d8;
                }
            }
        </style>

        <div class="vertical-line_3"></div>
    """, unsafe_allow_html=True)

with method_panel_col:

    st.markdown("")
    st.markdown("")
    st.markdown("<h3 style='text-align: center;'>方法</h3>", unsafe_allow_html=True)
    st.markdown("---")

    method_select_expander = st.expander("methods", expanded=True)

    with method_select_expander:

        # 4-3 选择方法
        method_selection = st.multiselect(
            label="分析方法（可多选）", key="create_method_selection",label_visibility='collapsed',
            options=list(algorithms.keys())
        )

        methods_direction = st.selectbox("分析方向", ["AB&BA", "AB", "BA"], key="methods_direction")

        method_output_button = st.button("输出", key='method_output_button', use_container_width=True)

        hr_second()

    # 4-4 方法的打印内容
    if method_selection:
        method_print = {m + 1: method_ for m, method_ in enumerate(method_selection)}
    else:
        method_print = {}

    if method_selection:

        with method_select_expander:
            st.markdown('')
            method_kwargs = param_controller(
                param_list=method_selection,
                para_descriptions=method_descriptions,
                param_controls=method_param_controls,cols=2,inner_cols=2,
                desc='方法'
            )


if check_file_button or check_file_with_DPLS_button or method_output_button or classify_output_button:

    print('check_file_button')

    use_files_dict = {}
    processed_files_dict = {}

    run_AB = db_select_kwargs['relation'] in ["AB", "AB&BA"]
    run_BA = db_select_kwargs['relation'] in ["BA", "AB&BA"]

    for database in database_selections:


        if use_data_type == use_data_options[1]:

            read_files, file_names, files_cause, file_description = data_presenter(database, **db_select_kwargs)
            raw_files_dict[database] = {"files_pair": dict(zip(file_names, read_files)),
                                     "files_cause": dict(zip(file_names, files_cause)),
                                    "description": file_description}

        else:

            pass


        if preprocess_selection:

            file_processed = {key: {} for key in raw_files_dict[database].keys()}

            if run_AB:

                for key, values in do_process(raw_files_dict[database], preprocess_selection).items():

                    if isinstance(values, dict):
                        file_processed[key] = file_processed[key] | values
                    else:
                        file_processed[key] = values


            if run_BA:

                for key, values in do_process(raw_files_dict[database], preprocess_selection, reverse=True).items():

                    if isinstance(values, dict):
                        file_processed[key] = file_processed[key] | values
                    else:
                        file_processed[key] = values

        else:

            file_processed = raw_files_dict[database]


        use_files_dict[database] = raw_files_dict[database]
        processed_files_dict[database] = file_processed

        files_cause = list(use_files_dict[database]['files_cause'].values())

        count_0 = files_cause.count(0)
        count_1 = files_cause.count(1)

        total_0 += count_0
        total_1 += count_1
        total_file += len(use_files_dict[database]['files_pair'])


    st.session_state[f"use_files_dict"] = use_files_dict
    st.session_state[f"processed_files_dict"] = processed_files_dict
    st.session_state['checking_file'] = None

    del raw_files_dict


# 模拟样本的独立功能区, 单独于 host_panel_check_file 外 ----------------------------------------------------------------------


with host_panel_file_detial:
    st.markdown("")
    st.markdown("")
    st.markdown(
        "<h3 style='text-align: center;'>读取的文件</h3>",
        unsafe_allow_html=True
    )
    hr_second(dark_color="#244690", height=2, light_color="#26519d")





if check_file_button or check_file_with_DPLS_button or processed_files_dict:

    if check_file_button:
        st.session_state['print_pred'] = False
    if check_file_with_DPLS_button:
        st.session_state['print_pred'] = True

    with host_panel_file_detial:

        with st.spinner("正在加载文件..."):

            st.session_state['checking_file'] = expand_raw_now_files(processed_files_dict, expand=True,
                                                                     total_file=total_file, block_id=use_data_options.index(use_data_type),
                                                                     description_type= 'latex' if use_data_options.index(use_data_type) == 0 else 'str',
                                                                     print_pred=st.session_state.get('print_pred',False),
                                                                     thread=Thread,
                                                                     checking_file=st.session_state['checking_file'])

# 3-4 预处理的打印内容
if preprocess_selection:
    preprocess_print = {p + 1: process_ for p, process_ in enumerate(preprocess_selection)}
else:
    preprocess_print = {}


if st.session_state.get("use_values_dict", False):

    use_values_dict = st.session_state["use_values_dict"]
    print("have_values")

else:

    use_values_dict = {}

    print("no values")


# Methods 值的计算器
def cal_values(file_pairs, reverse=False):


    # 默认第一列为原因
    if reverse != 1:
        reason = 0
        result = 1
    else:
        reason = 1
        result = 0


    file_processed_for_cal_values = do_process_one(file_pairs, preprocess_selection, reverse=reverse)

    df_list = []
    for m, method in enumerate(method_selection):

        if method in algorithms.keys():
            method_return = parallel_wrapper(func=algorithms[method], file_value_dict=file_processed_for_cal_values, desc=method,
                                             thread=Thread, reason=reason, result=result,
                                             **method_kwargs.get(method, {}),
                                             )
            method_return_DF = pd.DataFrame.from_dict(method_return, orient='index')
            named_method_DF = return_values_DF(method_return_DF, pre_process=preprocess_selection, method=method,
                                               reverse=reverse, **method_kwargs.get(method, {}))
            df_list.append(named_method_DF)
        else:
            pass


        # now_mission += 2
        # title_bar_progress.progress(now_mission / total_mission)

    try:

        result_df = pd.concat(df_list, axis=1)

    except ValueError:

        return None

    return result_df


with method_panel_file_detial:

    st.markdown("")
    st.markdown("")
    st.markdown("<h3 style='text-align: center;'>文件的方法值</h3>", unsafe_allow_html=True)
    st.markdown("---")


if method_output_button or classify_output_button:

    use_values_dict = {}

    method_run_AB = methods_direction in ["AB", "AB&BA"]
    method_run_BA = methods_direction in ["BA", "AB&BA"]

    for db_name, db_values in use_files_dict.items():

        db_cal_values = []

        if method_run_AB:

            db_cal_values.append(cal_values(db_values["files_pair"]))

        if method_run_BA:

            db_cal_values.append(cal_values(db_values["files_pair"], reverse=True))

        use_values_dict[db_name] = pd.concat(db_cal_values, axis=1)

    st.session_state["use_values_dict"] = use_values_dict


if use_values_dict:

    with method_panel_file_detial:

        with st.expander('方法值', expanded=True):
            st.dataframe(pd.concat(list(use_values_dict.values()), axis=0), height=1157)



if classify_output_button:

    classify_results = {}

    if use_values_dict:

        file_values = pd.concat(list(use_values_dict.values()), axis=0)

        for classify in classify_selection:

            classify_result = do_classify(classify, file_values=file_values, **classify_kwargs.get(classify, {}))
            classify_results[classify] = classify_result
    



st.markdown("")
st.markdown("")
hr_second(dark_color="#244690", height=2.1, light_color="#26519d")

summarize_panel_col, summarize_panel_sep, summarize_panel_file_detial= st.columns(
    [.496,.03, .8])

with summarize_panel_sep:
    st.markdown("""
        <style>
            .vertical-line_5 {
                width: 0.5px;
                height: 948px;
                margin: auto;
                margin-top: -5px;
            }

            @media (prefers-color-scheme: dark) {
                .vertical-line_5 {
                    background-color: #333333;
                }
            }

            @media (prefers-color-scheme: light) {
                .vertical-line_5 {
                    background-color: #d8d8d8;
                }
            }
        </style>

        <div class="vertical-line_5"></div>
    """, unsafe_allow_html=True)

# 9 在 detail 区打印参数 --------------------------------------------------------------------------------------------------

with summarize_panel_file_detial:
    st.markdown("")
    st.markdown("")
    st.markdown("<h3 style='text-align: center;'>总结</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("")

    detail_panel = st.columns([1, 1, 1, 1])

with summarize_panel_col:

    st.markdown("")
    st.markdown("")
    st.markdown("<h3 style='text-align: center;'>下载</h3>", unsafe_allow_html=True)
    st.markdown("---")

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


def del_cache():

    def clean_dict(d):
        return {
            k: v for k, v in d.items()
            if not isinstance(v, (np.ndarray, pd.DataFrame, DPLS))
        }

    st.session_state = clean_dict(st.session_state)

    if  'eg_created' in st.session_state:
        del st.session_state['eg_created']

    if "create_eg_kwargs" in st.session_state:
        del st.session_state["create_eg_kwargs"]

    if f"Cause_GUI_eg_created" in st.session_state:
        del st.session_state["Cause_GUI_eg_created"]

    if f"Cause_GUI_use_files_dict" in st.session_state:
        del st.session_state["Cause_GUI_use_files_dict"]

    if f"use_files_dict" in st.session_state:
        del st.session_state["use_files_dict"]

    if "Cause_GUI_clear_button" in st.session_state:
        del st.session_state["Cause_GUI_clear_button"]
        del st.session_state['classify_output_button']
        del st.session_state['preprocess_output_button']
        del st.session_state['method_output_button']


    if "add_x_b" in st.session_state:
        del st.session_state["add_x_b"]
        del st.session_state["minus_x_b"]
        del st.session_state["add_xtox_b"]
        del st.session_state["minus_xtox_b"]


# 项目文件记录器
def save_project(project_folder_path: str, project_name: str):

    # 先删除缓存
    del_cache()

    if os.path.exists(project_folder_path):
        pass
    else:
        os.makedirs(project_folder_path)

    run_project = st.session_state

    #添加一些版本信息
    run_project['input_project_name'] = project_name
    run_project['Version'] = now_version

    print(f"Cause_GUI_seed_value", st.session_state.get('Cause_GUI_seed_value', None))

    save_project_path = os.path.join(project_folder_path, f'{project_name}.json')

    with open(save_project_path, 'w', encoding='utf-8') as f:
        json.dump(run_project, f, ensure_ascii=False, indent=4)


if save_project_button:

    if now_project_name:

        save_project(project_dir, f'{now_project_name}')
        st.toast(f'已存为{now_project_name}.json')

    else:
        from datetime import datetime
        current_time = datetime.now().strftime("%H-%M")
        save_project(project_dir, f'{current_time}_run')
        st.toast(f'已存为{current_time}_run.json')


# with report_expander:
#
#     if "input_project_name" in st.session_state:
#
#         st.success(f"项目[{st.session_state['input_project_name']}]已载入")
#
# if project_input:
#     with report_expander:
#         st.success(f"项目{selected_project}已载入")


# def project_transfer():
#     # 预处理流程
#     preprocess_project = {
#         f"预处理步骤: {step}": preprocess_kwargs[step] for idx, step in enumerate(st.session_state.get("preprocess_selection", []), start=1)
#     }
#
#     # 方法参数
#     method_project = {
#         f"分析方法: {method}": method_kwargs[method] for idx, method in enumerate(st.session_state.get("create_method_selection", []), start=1)
#     }
#
#     # 分类参数
#     classify_project = {
#         f"分类方法: {clf}": classify_kwargs[clf] for idx, clf in enumerate(st.session_state.get("classify_selection", []), start=1)
#         if clf != "All"
#     }
#
#     # 合并所有参数
#     all_project = {}
#     all_project.update(file_param)
#     all_project.update(preprocess_project)
#     all_project.update(method_project)
#     all_project.update(classify_project)
#
#     st.info(all_project)


# # 11 运行按钮  -----------------------------------------------------------------------------------------------------------
#
#
# if run_button:
#
#     st.sidebar.markdown("")
#
#     # 点击分析后立即存储本次项目
#     del_cache()
#     save_project(project_dir, f'last_run_project')
#     st.info("本次运行项目已存储")
#
#     # 删除缓存
#
#     now_files_dict = copy.deepcopy(use_files_dict)
#
#     with title_bar:
#
#         # 插入 CSS 控制进度条外边距
#         st.markdown("""
#             <style>
#             div[data-testid="stProgress"] {
#                 margin-top: 0px;
#             }
#             </style>
#         """, unsafe_allow_html=True)
#
#         render_dataset_title("总进度", font_size=20)
#         # 添加带标签的进度条（Streamlit >= 1.25）
#         title_bar_progress = st.progress(0)
#
#
#     try:
#
#         # 分析方向控制
#
#         run_AB = methods_direction in ["AB", "AB&BA"]
#         run_BA = methods_direction in ["BA", "AB&BA"]
#
#         # 数据选择
#
#         all_results = {}
#
#         if method_selection:
#
#             total_mission = (run_BA + run_AB) * (2*len(method_selection) + len(preprocess_selection)) * len(use_files_dict) + len(classify_selection)
#             method_results = []
#             file_causes = []
#
#             for db_dict in use_files_dict.copy().values():
#
#                 db_y = pd.DataFrame.from_dict(db_dict['files_cause'], orient='index')
#                 file_causes.append(db_y)
#
#             y_df = pd.concat(file_causes, axis=0)
#
#             if run_AB:
#
#                 st.success("正在分析方向:AB")
#
#                 cal_values()
#
#                 AB_db_methods_results = []
#
#                 for db_dict in now_files_dict.values():
#
#                     AB_db_methods_results.append(db_dict['AB_pair_methods'])
#
#                 AB_methods_results = pd.concat(AB_db_methods_results, axis=0)
#                 method_results.append(AB_methods_results)
#
#
#             if run_BA:
#
#                 st.success("正在分析方向:BA")
#
#                 cal_values(reverse=True)
#
#                 BA_db_methods_results = []
#
#                 for db_dict in now_files_dict.values():
#                     BA_db_methods_results.append(db_dict['BA_pair_methods'])
#
#                 BA_methods_results = pd.concat(BA_db_methods_results, axis=0)
#                 method_results.append(BA_methods_results)
#
#             # 文件处理
#
#             # 汇总所有数据集, 所有方法的结果
#             method_values_df = pd.concat(method_results, axis=1)
#
#             st.success(f"Methods 分析完成!")
#             st.dataframe(method_values_df)
#
#             method_values_df_with_y = pd.concat([method_values_df, y_df], axis=1)
#
#             all_results[f'Values-results_{file_direction}.xlsx'] = method_values_df_with_y
#
#             if classify_selection:
#
#                 classify_results = {}
#                 num_rows_with_nan = method_values_df_with_y.isna().any(axis=1).sum()
#
#                 if num_rows_with_nan > 0:
#
#                     if illegal_compatible_mode:
#                         method_values_df_with_y = method_values_df_with_y.dropna(axis=0, how='any')
#                         with report_expander:
#                             st.warning(f"检测到[{num_rows_with_nan}]个非法样本, 已移除")
#                         st.warning(f"检测到[{num_rows_with_nan}]个非法样本, 已移除")
#                     else:
#                         with report_expander:
#                             st.warning(f"检测到[{num_rows_with_nan}]个非法样本, 若报错请尝试在[非法样本兼容模式]下分析")
#                         st.warning(f"检测到[{num_rows_with_nan}]个非法样本, 若报错请尝试在[非法样本兼容模式]下分析")
#
#                 train_list, test_lit = spliter(method_values_df_with_y.shape[0],
#                                                cv=classify_cv,
#                                                mode=cv_mode,
#                                                random_before=classify_shuffle_confirm,
#                                                shuffle_seed=classify_shuffle_seed,
#                                                )
#
#                 st.success('正在分类分析...')
#
#                 classify_bar = stqdm(classify_selection, total=len(classify_selection))
#                 for classify in classify_selection:
#                     classify_bar.set_description(f"正在执行:{classify}")
#                     classify_result = do_classify(classify, file_values=method_values_df_with_y,
#                                                   train_list=train_list, test_list=test_lit, **classify_kwargs.get(classify, {}))
#                     classify_results[classify] = classify_result
#                     classify_bar.update(1)
#
#
#                 classify_results_df = pd.DataFrame.from_dict(classify_results, orient='index')
#                 classify_results_df['Mean'] = classify_results_df.mean(axis=1)
#                 st.success('Classify 分析完成!')
#
#                 # 完成后事项 ---------------------------------------------------------------------------------------------
#
#                 with report_expander:
#                     st.success('Analysis completed')
#                 st.dataframe(classify_results_df)
#
#                 classify_file_name = f"Classify-results_{file_direction}"
#                 all_results[f"{classify_file_name}.xlsx"] = classify_results_df
#
#             zip_buffer = io.BytesIO()
#
#             with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
#                 for filename, df in all_results.items():
#                     # 每个 DataFrame 转换为 Excel 文件流
#                     excel_buffer = io.BytesIO()
#                     with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
#                         df.to_excel(writer, index=True)
#                     excel_buffer.seek(0)
#                     # 将该 Excel 写入 zip 包
#                     zip_file.writestr(filename, excel_buffer.read())
#
#             # 重要：把 ZIP 指针移到开头
#             zip_buffer.seek(0)
#
#             # 下载按钮
#             st.download_button(
#                 label="📦 下载所有结果 (ZIP)",
#                 data=zip_buffer,
#                 file_name=f"File_{Database_print}_{method_print}_{classify_print}_all.zip",
#                 mime="application/zip"
#             )
#
#             st.markdown('本次运行项目已保存, 点击下载按钮将重置页面')
#
#         else:
#
#             total_mission = (len(preprocess_selection)*len(use_files_dict)) if len(preprocess_selection) > 0 else 1
#             cal_values()
#
#         st.markdown('---')
#         st.markdown("<h3 style='text-align: center;'>抽样读取面板</h3>", unsafe_allow_html=True)
#         st.markdown('')
#
#         # 定义运行完毕后的抽样读取面板, 只能再定义一次, 复用 expand_raw_now_files() 位置会不对 -------------------------------------
#
#         with st.spinner('读取文件中...'):
#
#             with st.expander('抽样读取面板'):
#
#                 st.markdown('')
#                 st.markdown('---')
#
#                 check_file_panel_raw, check_file_panel_sep2, check_file_panel_now, check_file_panel_3, check_file_panel_y = st.columns(
#                     [1, 0.022, 1, 0.022, 0.6])
#
#                 with check_file_panel_raw:
#
#                     with st.expander('原始值', expanded=True):
#                         # 文件列表标题
#                         cols = st.columns([3, 4])
#                         cols[0].markdown("<div style='text-align: left; margin-top: 30px; padding-left:10px;'>文件名</div>",
#                                          unsafe_allow_html=True)
#                         cols[1].markdown(
#                             f"<div style='text-align: right; margin-top: 20px; padding-right: 20px;'> <span style='color: #55dd99; font-size:20px;'><strong>{total_file}</strong></span> files</div>",
#                             unsafe_allow_html=True)
#                         st.markdown("---")
#
#                         for db_name, db_values in use_files_dict.items():
#
#                             db_len = len(db_values["files_pair"])
#
#                             with st.expander(f'{db_name} | {db_len} files'):
#
#                                 for name, values in db_values["files_pair"].items():
#                                     with st.expander(f'{name} | {values.shape}'):
#                                         st.dataframe(values)
#
#                     st.markdown("---")
#
#                 with check_file_panel_now:
#
#                     with st.expander('Processed values', expanded=True):
#                         # 文件列表标题
#                         cols = st.columns([1.6, 4])
#                         cols[0].markdown("<div style='text-align: left; margin-top: 30px; padding-left:10px;'>文件名</div>",
#                                          unsafe_allow_html=True)
#
#                         cols[1].markdown(
#                             f"<div style='text-align: right; margin-top: 27px; padding-right: 20px;'> <span style='font-size:15px;'>{preprocess_print}</span></div>",
#                             unsafe_allow_html=True)
#                         st.markdown("---")
#
#                         for db_name_, db_values_ in now_files_dict.items():
#
#                             db_len = len(db_values_["files_pair"])
#
#                             with st.expander(f'{db_name_} | {db_len} files'):
#
#                                 for name, values in db_values_["files_pair"].items():
#                                     with st.expander(f'{name} | {values.shape}'):
#                                         st.dataframe(values)
#
#                     st.markdown("---")
#
#                 with check_file_panel_y:
#
#                     with st.expander('Cause directions', expanded=True):
#                         # 文件列表标题
#                         cols = st.columns([4, 4])
#
#                         cols[0].markdown(
#                             f"<div style='text-align: left; margin-top: 20px; padding-left: 20px;'> <span style='color: #4477dd; font-size:20px;'><strong>[{total_1}] </strong></span> A -> B</div>",
#                             unsafe_allow_html=True)
#
#                         cols[1].markdown(
#                             f"<div style='text-align: right; margin-top: 20px; padding-right: 20px;'> <span style='color: #dd4477; font-size:20px;'><strong>[{total_0}] </strong></span> B -> A</div>",
#                             unsafe_allow_html=True)
#                         st.markdown("---")
#
#                         for db_name, db_values in use_files_dict.items():
#                             db_len = len(db_values["files_cause"])
#
#                             with st.expander(f'{db_name} | {db_len} files'):
#                                 files_cause_df = pd.DataFrame.from_dict(db_values["files_cause"], orient='index')
#
#                                 st.dataframe(files_cause_df)
#
#                     st.markdown("---")
#
#         # base_dir = os.path.dirname(os.path.abspath(__file__))
#         # temp_now_file_path = os.path.join(base_dir, 'temp_now_file')
#         #
#         # if os.path.exists(temp_now_file_path):
#         #     pass
#         # else:
#         #     os.makedirs(temp_now_file_path)
#         #
#         # json_ready_dict = {
#         #     key: df.to_dict(orient="records")  # 每个 DataFrame 转字典
#         #     for key, df in now_files_dict.items()
#         # }
#         #
#         # with open(rf'{temp_now_file_path}\last_now_file_record.json', 'w', encoding='utf-8') as f:
#         #     json.dump(json_ready_dict, f, ensure_ascii=False, indent=4)
#
#
#     except Exception as e:
#
#         with report_expander:
#             st.error("❌ 分析出错, 详情见底部报告")
#
#         st.error(e)
#         msg = traceback.print_exc()
#         print(msg)



# 12 文件读取实现 ---------------------------------------------------------------------------------------------------------


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

