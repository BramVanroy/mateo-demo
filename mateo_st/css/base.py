import streamlit as st


def add_custom_base_style():
    st.write(
        """<style>
a {color: #09ab3b !important;}
a:hover {text-decoration: none;}

/* Special button alignment */
[data-testid="column"] + [data-testid="column"] {
    align-self: flex-end;
}
[data-testid="column"] + [data-testid="column"] button {
    padding: 7px 14px;
}

/*  Progress bar custom color */
.stProgress > div > div > div > div {
    background-color: #94bf89;
}
</style>
        """,
        unsafe_allow_html=True,
    )