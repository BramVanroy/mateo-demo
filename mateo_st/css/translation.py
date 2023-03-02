import streamlit as st


def add_custom_translation_style():
    st.markdown(
        """<style>
.translations-wrapper {
    border: 1px solid;
    border-radius: 0.25rem;
    padding: 0.48em 0.48em;
    margin-bottom: 1em;
}
</style>
      """,
        unsafe_allow_html=True,
    )
