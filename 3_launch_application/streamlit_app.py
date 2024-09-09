import os

import streamlit as st

from utils.neo4j_utils import is_neo4j_server_up, reset_neo4j_server

# Trigger Neo4j server in background.
if not is_neo4j_server_up():
    reset_neo4j_server()

cwd = os.getcwd()

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
pg = st.navigation(
    [
        st.Page(
            cwd + "/streamlit_pages/rag_app_page.py",
            title="Q/A for AI/ML research papers",
            icon=":material/description:",
        ),
        st.Page(
            cwd + "/streamlit_pages/model_selection_page.py",
            title="Model Selection",
            icon=":material/tv_options_edit_channels:",
        ),
        st.Page(
            cwd + "/streamlit_pages/knowledge_graph_visualisation_page.py",
            title="Knowledge Graph",
            icon=":material/hub:",
        ),
    ]
)
pg.run()
