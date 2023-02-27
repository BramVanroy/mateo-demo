import streamlit as st
from functions.utils import add_custom_base_style
from sections.evaluate.input_section import get_input_content
from sections.evaluate.metrics_section import get_metrics_content

from functions.utils import set_session_keys


def main():
    st.set_page_config(page_title="Evaluate | MATEO", page_icon="💯")
    add_custom_base_style()
    set_session_keys()
    st.title("💯 Evaluate")

    # Section: Input
    get_input_content()

    # Section: metrics
    get_metrics_content()



if __name__ == "__main__":
    main()
