from io import StringIO
from typing import Tuple

import streamlit as st
from mateo_st.metrics_constants import METRICS_META
from mateo_st.utils import isfloat, isint, load_css, set_general_session_keys


def _init():
    st.set_page_config(page_title="Evaluate Machine Translations | MATEO", page_icon="📏")
    load_css("base")

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

    # Iterate over all the metrics in METRIC_META and add each one and all its options
    for metric_idx, (ugly_metric_name, meta) in enumerate(METRICS_META.items()):
        metric_container = st.container()
        pretty_metric_name = meta.name
        metric_container.checkbox(f"Use {pretty_metric_name}", key=ugly_metric_name, value=meta.is_default_selected)

        # Add options as an "expander"
        if meta.options:
            expander = metric_container.expander(f"{meta.name} options")
            for opt_idx, opt in enumerate(meta.options):
                opt_name = opt.name
                has_choices = opt.choices

                opt_label = f"{ugly_metric_name}--{opt_name}"
                kwargs = {
                    "label": f"{opt_name} (default: '{opt.default}')",
                    "help": opt.description,
                    "key": opt_label,
                }

                # If this option is one with choices or with free input
                if has_choices:
                    expander.selectbox(options=opt.choices, index=opt.choices.index(opt.default), **kwargs)
                else:
                    # Type field is determined by the FIRST item in the list types
                    dtype = opt.types[0]
                    kwargs["value"] = opt.default
                    if dtype is str:
                        expander.text_input(**kwargs)
                    elif dtype in (int, float):
                        if opt.empty_str_is_none:
                            expander.text_input(**kwargs)
                        else:
                            expander.number_input(**kwargs, step=0.01 if float in opt.types else 1)
                    elif dtype is bool:
                        expander.checkbox(**kwargs)


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

    # Check whether any of the selected metrics require source input
    if any(
            meta.requires_source and name in st.session_state and st.session_state[name]
            for name, meta in METRICS_META.items()
    ):
        src_file = st.file_uploader("Source file (only needed for some metrics, like COMET)")
        st.session_state["src_segments"] = read_file(src_file)

    num_sys = st.number_input("How many systems do you wish to compare? (max. 3)", step=1, min_value=1, max_value=3)

    for sys_idx in range(1, num_sys + 1):
        sys_file = st.file_uploader(f"System #{sys_idx} file")
        if sys_file is not None:
            stringio = StringIO(ref_file.getvalue().decode("utf-8"))
            st.session_state["sys_segments"][sys_idx] = stringio.read().splitlines()
        else:
            st.session_state["sys_segments"][sys_idx] = []


def _validate_state() -> Tuple[bool, str]:
    source_required = any(
        meta.requires_source and name in st.session_state and st.session_state[name]
        for name, meta in METRICS_META.items()
    )
    can_continue = True
    msg = "Make sure that the following requirements are met:\n"

    # At least one metric must be selected
    if not any(name in st.session_state and st.session_state[name] for name, meta in METRICS_META.items()):
        msg += "- At least one metric must be selected\n"
        can_continue = False

    # Reference translations must be given
    if not st.session_state["ref_segments"]:
        msg += "- Reference translations must be given\n"
        can_continue = False

    if source_required:
        # Source translations must be given if source_required
        if not st.session_state["src_segments"]:
            msg += "- A source file must be added because a metric that requires it was selected\n"
            can_continue = False
        # Source segments must be the same number as reference segments
        elif len(st.session_state["src_segments"]) != len(st.session_state["ref_segments"]):
            msg += "- The source file must contain the same number of lines as the reference file\n"
            can_continue = False

    # At least one set of sys translations need to exist
    if not any(st.session_state["sys_segments"].values()):
        msg += "- Must have at least one set of system translations\n"
        can_continue = False
    else:
        for sys_idx, sys_segs in st.session_state["sys_segments"].items():
            # System segments must be the same number as reference segments
            if len(sys_segs) != len(st.session_state["ref_segments"]):
                msg += f"- Reference file #{sys_idx} must contain the same number of lines as the reference file\n"
                can_continue = False

    # Type-checking for options that have empty_str_is_none
    for name, meta in METRICS_META.items():
        if name in st.session_state and st.session_state[name]:
            for opt in meta.options:
                opt_name = opt.name
                if opt.empty_str_is_none:
                    session_opt = st.session_state[f"{name}--{opt_name}"]
                    # Collect tests for the specified dtypes
                    do_tests = []
                    for dtype in opt.types:
                        if dtype is float:
                            do_tests.append(isfloat)
                        elif dtype is int:
                            do_tests.append(isint)

                    # Check that the user input is indeed either an empty string or (one of) the expected dtypes
                    # Also check for special stringy float-types like "nan", "-inf", etc.
                    if (
                            (session_opt != "" and not any(do_test(session_opt) for do_test in do_tests))
                            or session_opt.lower().replace("-", "").replace("+", "") in ("inf", "infinity", "nan")
                    ):
                        msg += (f"- Option `{opt_name}` in {meta.name} must be one of: empty string,"
                                f" {', '.join([t.__name__ for t in opt.types])}\n")
                        can_continue = False

    st.session_state
    return can_continue, msg


def main():
    _init()
    _metric_selection()
    _data_input()
    can_continue, msg = _validate_state()

    msg_container = st.empty()
    if not can_continue:
        msg_container.warning(msg)
    else:
        msg_container.empty()
        if st.button("Calculate scores"):
            st.write(st.session_state)
        else:
            st.write("Click the button above to start calculating the automatic evaluation scores for your data")


if __name__ == "__main__":
    main()
