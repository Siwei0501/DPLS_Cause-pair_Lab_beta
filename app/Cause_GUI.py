
import importlib.util
import json
import os
import sys

import webbrowser
import stqdm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# 导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GUI_functions.casual_pair_tester import process, algorithms, return_values_DF
from GUI_modules.custom_GUI_module import *
from GUI_modules.create_data_module import create_data

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
#全局变量字典
page_param_dict = {}
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

    now_project_name_print = now_project_name if now_project_name else "Cause-pair Project"

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


use_data_options = ["模拟样本", "本地样本", "上传样本"]

use_data_type = st.sidebar.selectbox("**选择分析的数据种类**", options=use_data_options, key="use_data_type")

st.sidebar.markdown('---')
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
    pair_presented_data, col_names, files_causes_y, files_X, filex_y = module.return_cause_pair(**dataset_kwargs)  # 假设模块中有 some_function()

    description_path = os.path.join(local_data_dir, dataset_name, 'description.txt')

    try:
        with open(description_path, 'r', encoding='utf-8') as f:
            description =  f.read()
    except (FileNotFoundError, PermissionError) as e:
        description =  "No description"

    return pair_presented_data, col_names, files_causes_y, files_X, filex_y, description


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

else:

    use_files_dict = {}


if st.session_state.get(f"processed_files_dict", False):

    processed_files_dict = st.session_state.get(f"processed_files_dict")

else:

    processed_files_dict = {}


# 3-3 选择预处理
with host_panel_preprocess:

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

            page_param_dict.update(preprocess_kwargs)


def do_process_one(files_pair, preprocess_selection_, reverse=False):

    # 默认第一列为原因
    if reverse != 1:
        reason = 0
        result = 1
    else:
        reason = 1
        result = 0

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

print("use_data_typee", use_data_type )
print("use_data_type_in_sesstion", st.session_state["use_data_type"])
print("use_data_type == use_data_options[1]", use_data_type == use_data_options[1])

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
                                             key="local_database_selections", placeholder="本地数据集 (可多选)",
                                             options=list(local_datasets),
                                             default=list(local_datasets)[0], label_visibility='collapsed')

        st.markdown("---")

        data_sort_mode = st.selectbox(label='文件排序',
                                      key="local_data_sort_mode",
                                      options=["按首字母", "按大小"],
                                      )

        direction_panel = st.columns([1, 1])

        # 文件方向
        with direction_panel[0]:
            illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False], key="local_illegal_compatible_mode")
            file_direction = st.selectbox("样本方向", ["AB&BA", "AB", "BA", ], key="local_file_direction")

        # 分析方向
        with direction_panel[1]:
            DPLS_workspace = st.selectbox("**DPLS_workspace**", ['fusion', 'single'], key="local_DPLS_workspace")
            preprocess_direction = st.selectbox("预处理方向", ["AB&BA", "AB", "BA"], key="local_preprocess_direction")

        col_file_num, col_test_seed = st.columns([1, 1])

        with col_file_num:
            test_files_num = st.number_input(
                "文件抽样数",
                min_value=10,
                max_value=10000,
                value=50,
                step=10
                , key="local_test_files_num")

        with col_test_seed:
            seed_value = st.number_input(
                "文件抽样种子",
                min_value=0,
                max_value=20000,
                value=42,
                step=1
                , key="local_seed_value")

        thresh_range = st.slider("样本数限制在", 0, 10000, (100, 1500), key="local_thresh_range", step=100)

        check_file_button = st.button("**读取文件**", use_container_width=True)
        check_file_with_DPLS_button = st.button("**读取并预测 DPLS**", use_container_width=True)

    # 7-1-4 收集参数 -------------------------------------------------------------------------------------------------

    file_transfer_param = {

        "正在使用": use_data_type,
        "非法样本兼容": illegal_compatible_mode,
        '样本方向': file_direction,
        '预处理方向': preprocess_direction,
        '文件抽样数': test_files_num,
        '随机种子': seed_value,
        '样本量': thresh_range,

    }

    file_param = {

        "local_use_data_type": use_data_type,
        "local_illegal_compatible_mode": illegal_compatible_mode,
        'local_file_direction': file_direction,
        'local_preprocess_direction': preprocess_direction,
        'local_test_files_num': test_files_num,
        'local_seed_value': seed_value,
        'local_thresh_range': thresh_range,

    }

    page_param_dict.update(file_param)
    db_select_kwargs = {'relation': file_direction, 'threshold': list(thresh_range), 'seed': seed_value, 'test_SAMPLE': test_files_num}


elif use_data_type == use_data_options[0]:

    with host_next:

        raw_files_dict, file_param = create_data('Cause_GUI', thread=Thread)


    # 7-2-2 与 本地文件模式共用的功能部分 ----------------------------------------------------------------------------

    with host_panel_files:

        st.markdown("")
        st.markdown(
            "<h3 style='text-align: center;'>选择文件</h3>",
            unsafe_allow_html=True
        )
        hr_second(dark_color="#244690", height=2, light_color="#26519d")
        create_file_panel_expander = st.expander("**Created_files**", expanded=True)


    with create_file_panel_expander:

        if raw_files_dict:
            create_db_default = ["All"]

        else:

            create_db_default = []

        # 选择要分析的数据集
        database_selections = st.multiselect(label='模拟数据集 (可多选)',
                                             key="create_database_selections", placeholder="本地数据集 (可多选)",
                                             options=create_db_default + list(raw_files_dict.keys()),
                                             default=create_db_default, label_visibility='collapsed')

        if f"All" in database_selections:
            database_selections = list(raw_files_dict.keys())

        st.markdown("---")

        col_file_num, col_test_seed = st.columns([1, 1])

        with col_file_num:
            illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False], key="illegal_compatible_mode")
            test_files_num = st.number_input(
                "文件抽样数",
                min_value=10,
                max_value=file_param.get('test_files_num', 20),
                value=file_param.get('test_files_num', 20),
                step=10
                , key="create_test_files_num")

        with col_test_seed:
            DPLS_workspace = st.selectbox("**DPLS_workspace**", ['fusion', 'single'], key="create_DPLS_workspace")
            seed_value = st.number_input(
                "文件抽样种子",
                min_value=0,
                max_value=20000,
                value=42,
                step=1
                , key="create_seed_value")

        if create_db_default:
            st.markdown("")
            gui_success(f'接收到来自模拟器的 {len(raw_files_dict)} 个函数')
            st.markdown("---")

        check_file_button = st.button("**读取文件**", use_container_width=True)
        check_file_with_DPLS_button = st.button("**读取并预测 DPLS**", use_container_width=True)

    if not create_db_default:
        with report_zone:

            gui_warning('⚠️ 还未生成模拟数据, 请确认模拟数据参数后点击 [推送到项目]')

            st.markdown("---")


    file_transfer_param = {

        "正在使用": use_data_type,
        "非法样本兼容": illegal_compatible_mode,
        '文件抽样数': test_files_num,
        '随机种子': seed_value,

    }

    file_param.update({

        "create_use_data_type": use_data_type,
        "create_illegal_compatible_mode": illegal_compatible_mode,
        'create_test_files_num': test_files_num,
        'create_seed_value': seed_value,

    })

    page_param_dict.update(file_param)

    db_select_kwargs = {'relation': file_param.get('relation', "AB"), 'seed': seed_value, 'test_SAMPLE': test_files_num}


    # 7-3 定义[上传]数据参数控制面板 --------------------------------------------------------------------------------------------

elif use_data_type == use_data_options[2]:

    with host_panel_files:

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

        col_file_num, col_test_seed = st.columns([1, 1])

        with col_file_num:
            illegal_compatible_mode = st.selectbox("非法样本兼容", [True, False], key="upload_illegal_compatible_mode")
            test_files_num = st.number_input(
                "文件抽样数",
                min_value=0,
                max_value=len(uploaded_files_dict),
                value=len(uploaded_files_dict),
                step=10
                , key="upload_test_files_num")

        with col_test_seed:
            DPLS_workspace = st.selectbox("**DPLS_workspace**", ['fusion', 'single'], key="upload_DPLS_workspace")
            seed_value = st.number_input(
                "文件抽样种子",
                min_value=0,
                max_value=20000,
                value=42,
                step=1
                , key="upload_seed_value")

        thresh_range = st.slider("样本数限制在", 0, 10000, (100, 1500), key="upload_thresh_range", step=100)

        check_file_button = st.button("**读取**", use_container_width=True)
        check_file_with_DPLS_button = st.button("**读取并预测 DPLS**", use_container_width=True)


    file_transfer_param = {

        "正在使用": use_data_type,
        "非法样本兼容": illegal_compatible_mode,
        '文件抽样数': test_files_num,
        '随机种子': seed_value,
        '样本量': thresh_range,

    }

    file_param = {

        "upload_use_data_type": use_data_type,
        "upload_illegal_compatible_mode": illegal_compatible_mode,
        'upload_test_files_num': test_files_num,
        'upload_seed_value': seed_value,
        'upload_thresh_range': thresh_range,
    }

    page_param_dict.update(file_param)
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

    classify_param_dict = {}

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

    classify_param_dict = {
        "classify_selection": classify_selection,
        "cv_mode": cv_mode,
        "classify_shuffle_seed": classify_shuffle_seed,
        "classify_shuffle_confirm": classify_shuffle_confirm,
        "classify_cv": classify_cv
    }

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

        classify_param_dict.update(classify_kwargs)

    page_param_dict.update(classify_param_dict)

# 分类方法执行器
def do_classify(classify_name, file_values: pd.DataFrame, **kwargs):

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

    method_param_dict = {}

    with method_select_expander:

        # 4-3 选择方法
        method_selection = st.multiselect(
            label="分析方法（可多选）", key="create_method_selection",label_visibility='collapsed',
            options=list(algorithms.keys())
        )

        methods_direction = st.selectbox("分析方向", ["AB&BA", "AB", "BA"], key="methods_direction")

        method_output_button = st.button("输出", key='method_output_button', use_container_width=True)

        hr_second()

        method_param_dict["create_method_selection"] = method_selection
        method_param_dict["methods_direction"] = methods_direction

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

        method_param_dict.update(method_kwargs)


if check_file_button or check_file_with_DPLS_button or page_param_dict.get(f'Cause_GUI_pushed_create',False) or method_output_button or classify_output_button :


    use_files_dict = {}
    processed_files_dict = {}

    run_AB = db_select_kwargs['relation'] in ["AB", "AB&BA"]
    run_BA = db_select_kwargs['relation'] in ["BA", "AB&BA"]

    for database in database_selections:


        if use_data_type == use_data_options[1]:

            read_files, file_names, files_cause, db_X, db_y, file_description = data_presenter(database, **db_select_kwargs)
            raw_files_dict[database] = {"files_pair": dict(zip(file_names, read_files)),
                                     "files_cause": dict(zip(file_names, files_cause)),
                                    "description": file_description,
                                    "X": db_X, "y": db_y,
            }

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

            if DPLS_workspace == 'single':



                st.session_state['checking_file'] = single_workspace(processed_files_dict, expand=True,
                                                                         total_file=total_file, block_id=use_data_options.index(use_data_type),
                                                                         description_type= 'latex' if use_data_options.index(use_data_type) == 0 else 'str',
                                                                         print_pred=st.session_state.get('print_pred',False),
                                                                         thread=Thread,
                                                                         checking_file=st.session_state['checking_file'])

            elif DPLS_workspace == 'fusion':

                st.session_state['checking_file'] = fusion_workspace(processed_files_dict, total_file=total_file,
                                                                     block_id=use_data_options.index(use_data_type),
                                                                     description_type= 'latex' if use_data_options.index(use_data_type) == 0 else 'str',
                                                                     checking_file=st.session_state['checking_file'], thread=Thread)


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

    if st.session_state.get('method_param_dict', False):

        method_changed = (st.session_state['method_param_dict'] != method_param_dict)
        print("method_changed", method_changed)
    else:
        method_changed = True
        print("method_changed", method_changed)

    if method_changed:

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
        st.session_state['method_param_dict'] = method_param_dict.copy()

    else:
        pass


if use_values_dict:

    with method_panel_file_detial:

        with st.expander('方法值', expanded=True):

            method_print_DF = pd.concat(list(use_values_dict.values()), axis=0)
            # 合并 y 数据
            method_y_df = pd.concat(
                [pd.DataFrame.from_dict(db['files_cause'], orient='index', columns=['y']) for db in use_files_dict.values()],
                axis=0
            )
            method_print_DF = pd.concat([method_print_DF, method_y_df], axis=1)

            st.dataframe(method_print_DF, height = 1157)

classify_results = st.session_state.get("classify_results", {})
classify_results_DF = pd.DataFrame.from_dict(classify_results, orient='index') if classify_results else pd.DataFrame()


if classify_output_button:

    # 判断参数是否变化
    classify_changed = st.session_state.get('classify_param_dict') != classify_param_dict

    if classify_changed:

        files_values = pd.concat(list(use_values_dict.values()), axis=0)

        # 合并 y 数据
        y_df = pd.concat(
            [pd.DataFrame.from_dict(db['files_cause'], orient='index') for db in use_files_dict.values()],
            axis=0
        )
        files_values = pd.concat([files_values, y_df], axis=1)

        # 检查非法样本
        if classify_selection:
            num_rows_with_nan = files_values.isna().any(axis=1).sum()
            if num_rows_with_nan > 0:
                msg = f"检测到[{num_rows_with_nan}]个非法样本"
                if illegal_compatible_mode:
                    files_values = files_values.dropna(axis=0, how='any')
                    msg += ", 已移除"
                else:
                    msg += ", 若报错请尝试在[非法样本兼容模式]下分析"
                with report_zone:
                    st.warning(msg)
                st.warning(msg)

        # 分类执行
        with classify_panel_file_detial:
            classify_results = {}
            classify_bar = stqdm(classify_selection, total=len(classify_selection))
            for classify in classify_selection:
                classify_bar.set_description(f"正在执行:{classify}")
                classify_results[classify] = do_classify(
                    classify,
                    file_values=files_values,
                    **classify_kwargs.get(classify, {})
                )
                classify_bar.update(1)

            classify_results_DF = pd.DataFrame.from_dict(classify_results, orient='index')

        # 保存结果到 session_state
        st.session_state['classify_param_dict'] = classify_param_dict
        st.session_state['classify_results'] = classify_results

    else:

        pass


with classify_panel_file_detial:
    # 直接用已有结果
    st.dataframe(classify_results_DF)


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
    download_package_button = st.button("**封装页面数据**", use_container_width=True)
    # hr_second(dark_color="#244690", height=2.1, light_color="#26519d")


if st.session_state.get('download_dict', False):

    download_dict = st.session_state['download_dict']

else:

    download_dict = {

        'use_files_download': None,
        'project_download': None,
        'files_values_download': None,
        'classify_download': None,
    }


if download_package_button:


    download_dict = {

        'use_files_download': use_files_download_zip(use_files_dict, content_type="files_pair"),
        'project_download': None,
        'files_values_download': pd.concat(list(use_values_dict.values()), axis=0).to_csv().encode('utf-8') if use_values_dict else None,
        'classify_download':classify_results_DF.to_csv().encode('utf-8') if classify_results else None,
    }

    st.session_state['download_dict'] = download_dict
    st.session_state['last_package_session'] = page_param_dict
    st.rerun()

with summarize_panel_col:

    st.markdown("---")
    left_download_col,download_sep, right_download_col = st.columns([1,.08, 1])
    summarize_info_zone = st.container()
    with left_download_col:
        use_files_download_container = st.container()
        project_download_container = st.container()

    with right_download_col:
        method_download_container = st.container()
        classify_download_container = st.container()

with st.spinner("正在封装"):

    # 样本
    with use_files_download_container:
        st.markdown("")
        use_files_download_cols = st.columns([1, .22])
        st.markdown("---")

    with use_files_download_cols[0]:

        render_section_title("样本")

    with use_files_download_cols[1]:

        if download_dict['use_files_download']:

            st.download_button(
                label="🡇",
                data=download_dict['use_files_download'],
                file_name="use_files_download.zip",
                mime="application/zip",
                key="use_files_download_button"
            )
        else:

            use_files_download_button = st.button("－", key="use_files_download_button")

    # 项目
    with project_download_container:
        st.markdown("")
        project_download_cols = st.columns([1, .22])
        st.markdown("---")

    with project_download_cols[0]:
        render_section_title("项目")

    with project_download_cols[1]:
        project_download_button = st.button("🡇", key="project_download_button")


    # 方法
    with method_download_container:
        st.markdown("")
        method_download_cols = st.columns([1, .22])
        st.markdown("---")

    with method_download_cols[0]:
        render_section_title("方法")

    with method_download_cols[1]:

        if download_dict['files_values_download']:

            # 下载按钮
            st.download_button(
                label="🡇",
                data=download_dict['files_values_download'],
                file_name='files_values.csv',
                mime='text/csv'
            )

        else:

            use_files_download_button = st.button("－", key="use_values_download_button")


    # 分类
    with classify_download_container:
        st.markdown("")
        classify_download_cols = st.columns([1, .22])
        st.markdown("---")

    with classify_download_cols[0]:
        render_section_title("分类")

    with classify_download_cols[1]:

        if download_dict['classify_download']:

            # 下载按钮
            st.download_button(
                label="🡇",
                data=download_dict['classify_download'],
                file_name='classify_results.csv',
                mime='text/csv'
            )

        else:

            classify_download_button = st.button("－", key="classify_download_button")


with summarize_info_zone:


    if page_param_dict == st.session_state.get('last_package_session', False):

        gui_success('当前页面为最新')

    elif st.session_state.get('last_package_session', False):

        gui_warning('⚠️ 页面有改动, 封装可能非最新')

    else:

        gui_info('未封装的页面')

    st.markdown("---")



with detail_panel[0]:

    render_section_title('Samples', font_size=20)
    with st.expander('samples', expanded=True):
        st.markdown('---')
        display_detial_dict(file_transfer_param)
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

    state_to_save = {
        k: v for k, v in st.session_state.items()
        if not (
                (isinstance(k, str) and (k.endswith('button') or k.endswith('dict')))
                or isinstance(v, (np.ndarray, pd.DataFrame, DPLS))  # 若要包含 DPLS，添加到这个元组里
        )
    }

    print('state_to_save', list(state_to_save.keys()))


    if  'eg_created' in state_to_save:
        del state_to_save['eg_created']

    if "create_eg_kwargs" in state_to_save:
        del state_to_save["create_eg_kwargs"]

    if f"Cause_GUI_eg_created" in state_to_save:
        del state_to_save["Cause_GUI_eg_created"]

    return state_to_save


# 项目文件记录器

def save_project(project_folder_path: str, project_name: str):

    # 先删除缓存
    run_project = del_cache()

    print('run_project', list(run_project.keys()))

    if os.path.exists(project_folder_path):
        pass
    else:
        os.makedirs(project_folder_path)

    #添加一些版本信息
    run_project['input_project_name'] = project_name
    run_project['Version'] = now_version

    print(f"now_project_name_save_before", st.session_state.get('now_project_name', None))

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
if project_input:
    st.toast(f'已读取项目: {selected_project[:-5]}')


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

