
import streamlit as st

pages = {

    "实验室":[st.Page('Cause_GUI.py', title="Cause-pair Lab", icon="🔵"),
                      st.Page('DPLS_Check_GUI.py', title="DPLS check Lab", icon="🟡")
                      ],
    "设置":[st.Page('DPLS_GUI.py', title='DPLS 设置', icon="⚙️")]

}

pg = st.navigation(pages)
pg.run()

