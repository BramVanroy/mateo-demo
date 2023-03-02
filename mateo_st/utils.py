import streamlit as st
from streamlit.components.v1 import html

from translator import TRANS_LANG2KEY, TRANS_SIZE2MODEL, DEFAULT_BATCH_SIZE, DEFAULT_MODEL_SIZE


def update_lang(side: str):
    st.session_state[f"{side}_lang_key"] = TRANS_LANG2KEY[st.session_state[f"{side}_lang"]]
    if "translator" in st.session_state and st.session_state["translator"]:
        if side == "src":
            st.session_state["translator"].set_src_lang(st.session_state["src_lang"])
        else:
            st.session_state["translator"].set_tgt_lang(st.session_state["tgt_lang"])


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


def set_general_session_keys():
    if "src_lang" not in st.session_state:
        st.session_state["src_lang"] = "English"
    elif st.session_state["src_lang"] not in TRANS_LANG2KEY:
        st.session_state["src_lang"] = "English"

    if "src_lang_key" not in st.session_state:
        st.session_state["src_lang_key"] = TRANS_LANG2KEY[st.session_state["src_lang"]]

    if "tgt_lang" not in st.session_state:
        st.session_state["tgt_lang"] = "Dutch"
    elif st.session_state["tgt_lang"] not in TRANS_LANG2KEY:
        st.session_state["tgt_lang"] = "Dutch"

    if "tgt_lang_key" not in st.session_state:
        st.session_state["tgt_lang_key"] = TRANS_LANG2KEY[st.session_state["tgt_lang"]]

    if "mt_text" not in st.session_state:
        st.session_state["mt_text"] = ""

    if "other_hyps" not in st.session_state:
        st.session_state["other_hyps"] = []


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_cli_args():
    import argparse
    cparser = argparse.ArgumentParser()
    cparser.add_argument("--transl_no_cuda", action="store_true", help="whether to disable CUDA for translation")
    cparser.add_argument("--transl_batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="batch size for translating")
    cparser.add_argument("--transl_model_size", choices=list(TRANS_SIZE2MODEL.keys()), default=DEFAULT_MODEL_SIZE,
                         help="model size to use")
    return cparser.parse_args()
