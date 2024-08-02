import os

import streamlit as st

import utils.constants as const
import pgs.commons as st_commons

cwd = os.getcwd()

remote_model_endpoint, remote_model_id, remote_model_api_key = "", "", ""
is_remote_llm = False

st.header("Knowledge Graph based RAG Pipeline")
st.subheader(f"Select the {const.local_model_to_be_quantised.split('/')[1]} variant to use")

st.markdown(
    f"""
    The app is designed to use the
    [{const.local_model_to_be_quantised.split('/')[1]}](https://huggingface.co/{const.local_model_to_be_quantised}), 
    a language model that has been fine-tuned on a variety of tasks.
    You can choose to run the 4-bit quantised model in-session using local GPU,
    or choose to run a remote model compatible with [Open AI API](https://platform.openai.com/docs/api-reference/introduction).
    In case of remote LLM, you will need to provide the model endpoint, model ID and your API key.
    """
)

def choose_llm_action():
    # Access the global variables.
    global is_remote_llm
    global remote_model_endpoint, remote_model_id, remote_model_api_key
    if not is_remote_llm:
        # Load and cache the model for future use.
        st_commons.get_cached_local_model()
        st.session_state[st_commons.StateVariables.IS_REMOTE_LLM.value] = False
    else:
        st.session_state[st_commons.StateVariables.REMOTE_MODEL_ENDPOINT.value] = remote_model_endpoint
        st.session_state[st_commons.StateVariables.REMOTE_MODEL_ID.value] = remote_model_id
        st.session_state[st_commons.StateVariables.REMOTE_MODEL_API_KEY.value] = remote_model_api_key
        st.session_state[st_commons.StateVariables.IS_REMOTE_LLM.value] = True

llm_choice = st.radio(
    "Choose LLM type:",
    options=["Local LLM", "Remote LLM"],
)
is_remote_llm = llm_choice == "Remote LLM"

if is_remote_llm:
    st.markdown(f":gray[{st_commons.remote_llm_text}]")
    remote_model_endpoint = st.text_input('Model Endpoint:')
    remote_model_id = st.text_input('Model ID:')
    remote_model_api_key = st.text_input('API Key:', type='password')
else:
    st.markdown(f":gray[{st_commons.local_llm_text}]")

llm_chosen = st.button('Apply preferences and continue to application', on_click=choose_llm_action)
if llm_chosen:
    st.switch_page(cwd+"/pgs/rag_app.py")
