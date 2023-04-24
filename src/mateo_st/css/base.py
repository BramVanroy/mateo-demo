import streamlit as st


def add_custom_base_style():
    st.write(
        """<style>
a {color: #09ab3b !important;}
a:hover {text-decoration: none;}

/*  Progress bar custom color */
.stProgress > div > div > div > div {
    background-color: #94bf89;
}
</style>
        """,
        unsafe_allow_html=True,
    )
