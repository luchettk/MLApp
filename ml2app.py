import streamlit as st 

about_page = st.Page("about.py", title="About", icon=":material/home:")
data_page = st.Page("data.py", title="Data", icon=":material/edit:")

pg = st.navigation([about_page, data_page])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()