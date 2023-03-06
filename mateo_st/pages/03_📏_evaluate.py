import streamlit as st
from css import add_custom_base_style
from utils import get_cli_args, set_general_session_keys


def _init():
    st.set_page_config(page_title="Evaluate | MATEO", page_icon="💯")
    add_custom_base_style()

    set_general_session_keys()
    args = get_cli_args()

    # ... set undefined session state vars here

    st.title("📏 Evaluate")
    st.markdown(
        "First specify the metrics and metric options to use, and then upload your data."
    )


def main():
    _init()


if __name__ == "__main__":
    main()
