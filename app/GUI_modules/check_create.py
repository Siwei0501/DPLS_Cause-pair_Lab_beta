import streamlit as st

pages = {

    "实验室":[st.Page('create_data_module.py', title="🔵 create_data_module"),
                      st.Page('create_func_module.py', title="🟡 create_func_module")
                      ],
    "设置":[st.Page(rf'C:\Users\19012\OneDrive\Programs\Cause_GUI\app\DPLS_GUI.py', title='DPLS 设置')]

}

pg = st.navigation(pages)
pg.run()