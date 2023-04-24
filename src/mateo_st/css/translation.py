import streamlit as st


def add_custom_translation_style():
    st.markdown(
        """<style>
[data-testid="column"] + [data-testid="column"] {
    align-self: flex-end;
    text-align: center;
}
[data-testid="column"] + [data-testid="column"] button {
    padding: 7px 14px;
}
</style>
      """,
        unsafe_allow_html=True,
    )
