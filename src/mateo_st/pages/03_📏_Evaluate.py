from io import StringIO

import streamlit as st
from mateo_st.css import add_custom_base_style
from mateo_st.metrics_constants import METRICS_META
from mateo_st.utils import set_general_session_keys


def _init():
    st.set_page_config(page_title="Evaluate Machine Translations | MATEO", page_icon="📏", layout="wide")
    add_custom_base_style()

    set_general_session_keys()

    if "ref_segments" not in st.session_state:
        st.session_state["ref_segments"] = []

    if "src_segments" not in st.session_state:
        st.session_state["src_segments"] = []

    if "sys_segments" not in st.session_state:
        st.session_state["sys_segments"] = dict()

    st.title("📏 Evaluate")
    st.markdown("First specify the metrics and metric options to use, and then upload your data.")


def _metric_selection():
    st.markdown("## ✨ Metric selection")
    col1, col2 = st.columns(2, gap="medium")

    # Iterate over all the metrics in METRIC_META and add it and all possible options
    for metric_idx, (name, meta) in enumerate(METRICS_META.items()):
        # Artificially separate metrics over two columns
        col = col1 if metric_idx % 2 == 0 else col2

        metric_container = col.container()
        metric_name = meta["name"]
        chk_col, name_col = metric_container.columns([1, 5])
        chk_col.checkbox(f"include_{name}", label_visibility="hidden", help=f"Use {metric_name}?", key=name)
        name_col.markdown(f"**{metric_name}**")

        if "options" in meta and meta["options"]:
            expander = metric_container.expander(f"{meta['name']} options")
            for opt_idx, (opt_name, opt) in enumerate(meta["options"].items()):
                has_choices = "choices" in opt

                opt_name_col, opt_desc_col, opt_input_col = expander.columns((2, 4, 4))
                opt_name_col.write(opt_name)
                opt_desc_col.write(opt["description"])

                opt_label = f"{metric_name}--{opt_name}"
                if has_choices:
                    opt_input_col.selectbox(
                        opt_label,
                        options=opt["choices"],
                        index=opt["choices"].index(opt["default"]),
                        label_visibility="hidden",
                        help=opt["description"],
                        key=opt_label,
                    )
                else:
                    dtype = opt["type"]
                    force_str = "force_str" in opt and opt["force_str"]
                    if dtype is str or ((dtype is int or dtype is float) and force_str):
                        opt_input_col.text_input(
                            opt_label,
                            label_visibility="hidden",
                            value=opt["default"],
                            help=opt["description"],
                            key=opt_label,
                        )
                    elif dtype is int or dtype is float:
                        opt_input_col.number_input(
                            opt_label,
                            label_visibility="hidden",
                            value=opt["default"],
                            help=opt["description"],
                            key=opt_label,
                        )
                    elif dtype is bool:
                        opt_input_col.checkbox(
                            opt_label,
                            label_visibility="hidden",
                            value=opt["default"],
                            help=opt["description"],
                            key=opt_label,
                        )

                if opt_idx < len(meta["options"]) - 1:
                    expander.divider()

        if metric_idx < len(METRICS_META) - 1:
            col.divider()


def _data_input():
    st.markdown("## Input data 📄")
    st.write(
        "Add a reference file and one or more files with translations. One line per file."
        " Cannot contain empty lines and must be in UTF8!"
    )

    def read_file(uploaded_file):
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            return stringio.read().splitlines()
        else:
            return []

    ref_file = st.file_uploader("Reference file")
    st.session_state["ref_segments"] = read_file(ref_file)

    if any(
        meta["requires_source"] and name in st.session_state and st.session_state[name]
        for name, meta in METRICS_META.items()
    ):
        src_file = st.file_uploader("Source file (only needed for some metrics, like COMET)")
        st.session_state["ref_segments"] = read_file(src_file)

    num_sys = st.number_input("How many systems do you wish to compare? (max. 3)", step=1, min_value=1, max_value=3)

    for sys_idx in range(1, num_sys + 1):
        sys_file = st.file_uploader(f"System #{sys_idx} file")
        if sys_file is not None:
            stringio = StringIO(ref_file.getvalue().decode("utf-8"))
            st.session_state["sys_segments"][sys_idx] = stringio.read().splitlines()
        else:
            st.session_state["sys_segments"][sys_idx] = []


def main():
    _init()
    _metric_selection()
    _data_input()


if __name__ == "__main__":
    main()
