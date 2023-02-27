import streamlit as st
from streamlit.components.v1 import html

from functions.translator import TRANS_LANG2KEY


def update_lang(side: str):
    st.session_state[f"{side}_lang_key"] = TRANS_LANG2KEY[st.session_state[f"{side}_lang"]]
    if "translator" in st.session_state and st.session_state["translator"]:
        if side == "src":
            st.session_state["translator"].set_src_lang(st.session_state["src_lang"])
        else:
            st.session_state["translator"].set_tgt_lang(st.session_state["tgt_lang"])


def add_custom_base_style():
    st.write(
        """<style>
a {color: #09ab3b !important;}
a:hover {text-decoration: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_custom_input_style():
    # Styling to make the remove buttons prettier
    st.write(
        """<style>
        [data-testid="column"] + [data-testid="column"] {
            align-self: flex-end;
        }
        [data-testid="column"] + [data-testid="column"] button {
            padding: 10px 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_custom_pbar_style():
    # Add custom color for progressbar (same as in .streamlit/config.toml)
    st.markdown(
        """
    <style>
    .stProgress > div > div > div > div {
        background-color: #94bf89;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def add_custom_charcut_style():
    st.markdown(
        """
  <style>
    #charcut-table, #charcut-table td, #charcut-table th {border-spacing: 0;}
    #charcut-table th {padding: 10px;}
    #charcut-table td {padding: 5px;}
    #charcut-table th {border-top: solid black 2px; font-weight: normal;}
    #charcut-table #key {margin-left: auto; margin-right: auto; text-align: left;}
    #charcut-table .tophead {border-bottom: solid black 1px;}
    #charcut-table .src {font-style: oblique;}
    #charcut-table .trg {font-family: Consolas, monospace;}
    #charcut-table .del {font-weight: bold; color: #f00000;}
    #charcut-table .ins {font-weight: bold; color: #0040ff;}
    #charcut-table .shift {font-weight: bold;}
    #charcut-table .match {}
    #charcut-table .mainrow {border-top: solid black 1px; padding: 1em;}
    #charcut-table .srcrow {padding-bottom: 15px;}
    #charcut-table .compcol {border-left: solid lightgray 2px;}
    #charcut-table .seghead {color: gray; padding-bottom: 0;}
    #charcut-table .score {font-family: Consolas, monospace; font-size: large; text-align: center;}
    #charcut-table .detail {font-size: xx-small; color: gray; text-align: right;}
    #charcut-table .info {font-size: xx-small; color: gray;}
    #charcut-table .hover {background-color: yellow;}
  </style>
    """,
        unsafe_allow_html=True,
    )


def add_custom_charcut_scripts():
    html(
        """
<script>
    const doc = window.parent.document;
    function enter(segCls) {
      const els = doc.querySelectorAll(`.${segCls}`);
      for (const el of els) {
          el.classList.add("hover");
      }
    }
    function leave(segCls) {
      const els = doc.querySelectorAll(`.${segCls}`);
      for (const el of els) {
          el.classList.remove("hover");
      }
    }
    
    doc.querySelectorAll("#charcut-table .compcol > span").forEach(el => {
        const classes = [...el.classList];
        const segCls = classes.filter((c) => c.startsWith("seg"))[0];
        el.addEventListener("mouseenter", evt => {
            enter(segCls);
        });
        el.addEventListener("mouseleave", evt => {
            leave(segCls);
        });
    });
</script>""",
        width=0,
        height=0,
    )


COLORS_PLOTLY = {
    "default": ["#94bf89", "#BF8480", "#6A5E73"],
    # Add other styles if wanted
}


def add_custom_metrics_style():
    st.markdown(
        """
    <style>
    .metric header p {font-weight: bold;}
    .metric .metric-paper p::before {content: "📝"; margin-right: 0.36em;}
    .metric .metric-implementation p::before {content: "💻"; margin-right: 0.36em;}
    .metrics-wrapper {
        display: flex;
        flex-wrap: wrap;
        align-items: flex-start;
        gap: 1.5em;
    }
    /*Disable automatic anchors for titles on the metrics page. */
    .metric a[href^="#"] {opacity: 0; pointer-events: none;}
    .metric {
        border-radius: 6px;
        box-shadow: 0 2px 6px rgb(0 0 0 / 5%);
        padding: 0em 1em;
        border: 1px solid #f5f5f5;
        background: linear-gradient(45deg, #fbfbfb, white);
        position: relative;
        overflow: hidden;
        min-width: 300px;
        max-width: 100%;
        flex: 1;
    }
    .metric::before {
        content: "💡";
        position: absolute;
        top: 0;
        right: 0;
        font-size: 12em;
        opacity: 0.05;
        line-height: 1;
        transform: translate(25%, -15%);
    }
    .metric.neural::before {
        content: "🚀";
    }
    .metric .metric-langs {margin-bottom: 1em;}
    .metric .metric-langs details {
        border: 1px solid #efefef;
        box-shadow: 0 1px 3px rgb(0 0 0 / 3%);
        background: linear-gradient(45deg, #fbfbfb, white);
        border-radius: 6px;
        padding: .5em .5em 0;
    }
    
    .metric .metric-langs summary {
        font-weight: bold;
        margin: -.5em -.5em 0;
        padding: .5em;
    }
    
    .metric .metric-langs details[open] {
        padding: .5em;
    }
    
    .metric .metric-langs details[open] summary {
        margin-bottom: .5em;
    }
    </style>
      """,
        unsafe_allow_html=True,
    )


def set_session_keys():
    if "translator" not in st.session_state:
        st.session_state["translator"] = None

    if "no_cuda" not in st.session_state:  # Can be set via the command-line. Parsed in 01__mateo .
        st.session_state["no_cuda"] = False

    if "src_lang" not in st.session_state:
        st.session_state["src_lang"] = "English"
    elif st.session_state["src_lang"] not in TRANS_LANG2KEY:
        st.session_state["src_lang"] = "English"

    if "src_lang_key" not in st.session_state:
        st.session_state["src_lang_key"] = TRANS_LANG2KEY[st.session_state["src_lang"]]

    if "tgt_lang" not in st.session_state:
        st.session_state["tgt_lang"] = "Dutch; Flemish"
    elif st.session_state["tgt_lang"] not in TRANS_LANG2KEY:
        st.session_state["tgt_lang"] = "Dutch; Flemish"

    if "tgt_lang_key" not in st.session_state:
        st.session_state["tgt_lang_key"] = TRANS_LANG2KEY[st.session_state["tgt_lang"]]

    if "mt_text" not in st.session_state:
        st.session_state["mt_text"] = ""

    if "other_hyps" not in st.session_state:
        st.session_state["other_hyps"] = []
