import streamlit as st
from functions.utils import add_custom_base_style
from sections.evaluate.translation_model_section import get_translation_model_content

from functions.utils import set_session_keys


def main():
    st.set_page_config(page_title="Translate | MATEO", page_icon="💯")
    add_custom_base_style()
    set_session_keys()
    st.title("💯 Translate")

    # Section: Translation model
    get_translation_model_content()



if __name__ == "__main__":
    main()
