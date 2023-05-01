import warnings
from io import StringIO
from typing import List, Optional, Tuple, Dict, Union

import evaluate
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from evaluate import EvaluationModule
from mateo_st.metrics_constants import METRICS_META, postprocess_result, MetricMeta, MetricOption
from mateo_st.utils import (
    cli_args,
    create_download_link,
    isfloat,
    isint,
    load_css,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)


def _init():
    st.set_page_config(page_title="Evaluate Machine Translations | MATEO", page_icon="📏")
    load_css("base")

    if "ref_segments" not in st.session_state:
        st.session_state["ref_segments"] = []

    if "ref_file" not in st.session_state:
        st.session_state["ref_file"] = None

    if "src_segments" not in st.session_state:
        st.session_state["src_segments"] = []

    if "src_file" not in st.session_state:
        st.session_state["src_file"] = None

    if "sys_segments" not in st.session_state:
        st.session_state["sys_segments"] = dict()

    if "sys_files" not in st.session_state:
        st.session_state["sys_files"] = dict()

    if "metrics" not in st.session_state:
        st.session_state["metrics"] = dict()

    if "results" not in st.session_state:
        st.session_state["results"] = dict()

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
                has_choices = bool(opt.choices)

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
    st.markdown("## 📄 Input data")
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

    # Check whether any of the selected metrics require source input
    # If so, use a two-col layout for the input buttons, if not just use full-width reference input
    if any(
        meta.requires_source and name in st.session_state and st.session_state[name]
        for name, meta in METRICS_META.items()
    ):
        ref_inp_col, src_inp_col = st.columns(2)
        ref_file = ref_inp_col.file_uploader("Reference file")
        src_file = src_inp_col.file_uploader("Source file")
        st.session_state["src_segments"] = read_file(src_file)
        st.session_state["src_file"] = src_file.name if src_file else None
    else:
        st.session_state["src_segments"] = []
        st.session_state["src_file"] = None
        ref_file = st.file_uploader("Reference file")

    st.session_state["ref_segments"] = read_file(ref_file)
    st.session_state["ref_file"] = ref_file.name if ref_file else None

    max_sys = cli_args().eval_max_sys
    num_sys = st.number_input(
        "How many systems do you wish to compare? (max. 3)", step=1, min_value=1, max_value=max_sys
    )

    # Iterate over i..max_value. Reason is that we need to delete sys_idx if it does not exist anymore
    # This can happen when a user first has three systems and then changes it back to 1
    for sys_idx in range(1, max_sys + 1):
        if sys_idx <= num_sys:
            sys_file = st.file_uploader(f"System #{sys_idx} file")
            st.session_state["sys_segments"][sys_idx] = read_file(sys_file)
            st.session_state["sys_files"][sys_idx] = sys_file.name if sys_file else None
        else:
            if sys_idx in st.session_state["sys_segments"]:
                del st.session_state["sys_segments"][sys_idx]

            if sys_idx in st.session_state["sys_files"]:
                del st.session_state["sys_files"][sys_idx]


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
                if opt.choices:  # do not check for multi-select
                    continue

                if opt.empty_str_is_none:
                    opt_name = opt.name
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
                        session_opt != "" and not any(do_test(session_opt) for do_test in do_tests)
                    ) or session_opt.lower().replace("-", "").replace("+", "") in ("inf", "infinity", "nan"):
                        msg += (
                            f"- Option `{opt_name}` in {meta.name} must be one of: empty string,"
                            f" {', '.join([t.__name__ for t in opt.types])}\n"
                        )
                        can_continue = False

    return can_continue, msg


def _add_metrics_selection_to_state():
    """Solidify the selected metrics and their options in one easy-to-use dictionary
    that we can use later on to initialize the metrics.
    """
    for metric_name, meta in METRICS_META.items():
        if metric_name in st.session_state and st.session_state[metric_name]:
            st.session_state["metrics"][metric_name] = {}
            for opt in meta.options:
                opt_name = opt.name
                opt_val = st.session_state[f"{metric_name}--{opt_name}"]
                if opt.empty_str_is_none:
                    if opt_val == "":
                        opt_val = None
                    elif isinstance(opt_val, str):
                        if float in opt.types and isfloat(opt_val):
                            opt_val = float(opt_val)
                        elif int in opt.types and isint(opt_val):
                            opt_val = int(opt_val)

                st.session_state["metrics"][metric_name][opt_name] = opt_val


@st.cache_resource(show_spinner=False, max_entries=24, ttl=86400)
def _load_metric(metric_name: str, config_name: Optional[str] = None) -> EvaluationModule:
    """Load an individual metric
    :param metric_name: metric name
    :param config_name: optional config
    :return: loaded metric
    """
    return evaluate.load(metric_name, config_name=config_name)


@st.cache_data(show_spinner=False, ttl=86400)
def _compute_metric(
    metric_name: str,
    *,
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
    config_name: Optional[str] = None,
    **kwargs,
):
    metric = _load_metric(metric_name, config_name)
    if sources:
        return metric.compute(predictions=predictions, references=references, sources=sources, **kwargs)
    else:
        return metric.compute(predictions=predictions, references=references, **kwargs)


def _compute_metrics():
    results = {}
    pbar_text_ct = st.empty()
    pbar = st.progress(0)
    increment = 100 // (len(st.session_state["sys_segments"]) * len(st.session_state["metrics"]))
    progress = 0

    for sys_idx, sys_segs in st.session_state["sys_segments"].items():
        results[sys_idx] = {}
        for metric_evaluate_name, opts in st.session_state["metrics"].items():
            opts: Dict[str, Union[MetricOption, MetricMeta]] = opts.copy()  # Copy to not pop globally
            meta = METRICS_META[metric_evaluate_name]

            if sys_idx == 1:
                msg = f"(Down)loading metric <code>{meta.name}</code> and evaluating system #{sys_idx}."
                if meta.metric_class == "neural":
                    msg += f"<br><code>{meta.name}</code> is a neural metric, so downloading and calculating may take" \
                           f" a long while, depending on the size of your dataset."
            else:
                msg = f"Evaluating system #{sys_idx} with <code>{meta.name}</code>"
                if meta.metric_class == "neural":
                    msg += f"<br><code>{meta.name}</code> is a neural metric, so calculating may take" \
                           f" a long while, depending on the size of your dataset."

            pbar_text_ct.markdown(f'<p style="font-size: 0.8em">{msg}</code></p>', unsafe_allow_html=True)

            result = _compute_metric(
                metric_evaluate_name,
                predictions=sys_segs,
                references=st.session_state["ref_segments"],
                sources=st.session_state["src_segments"] if meta.requires_source else None,
                config_name=opts.pop("config_name", None),
                **opts,
            )

            result = postprocess_result(metric_evaluate_name, result)

            results[sys_idx][metric_evaluate_name] = {
                "corpus": result[meta.corpus_score_key],
                "sentences": result[meta.sentences_score_key] if meta.sentences_score_key else None,
            }

            progress += increment
            pbar.progress(progress)

    pbar_text_ct.empty()
    pbar.empty()
    st.session_state["results"] = results


def _build_corpus_df():
    res = st.session_state["results"]

    data = []
    for sys_idx, results in res.items():
        sys_data = {"system": st.session_state["sys_files"][sys_idx]}
        for metric_name, metric_res in results.items():
            sys_data[metric_name] = metric_res["corpus"]
        data.append(sys_data)

    df = pd.DataFrame(data)

    # Remove "sacre" (bleu's output with sacrebleu is "sacrebleu")
    df = df.rename(mapper=lambda col: col.replace("sacre", ""), axis=1)
    return df


def _draw_corpus_scores(df):
    df = df.rename(
        mapper=lambda col: f"{col} {'↑' if METRICS_META[col].higher_better else '↓'}" if col in METRICS_META else col,
        axis=1,
    )
    # Reshape DataFame for plotting
    df_melt = pd.melt(df, id_vars="system", var_name="metric", value_name="score")
    fig = px.bar(
        df_melt,
        x="metric",
        y="score",
        color="system" if len(st.session_state["sys_files"]) > 1 else None,
        barmode="group",
        template="plotly"
    )

    st.plotly_chart(fig)


def _style_df_for_display(df):
    rounded_df = df.replace(pd.NA, np.nan)
    rounded_df.iloc[:, 1:] = rounded_df.iloc[:, 1:].astype(float).round(decimals=2).copy()
    numeric_col_names = rounded_df.columns[1:].tolist()
    styled_df = rounded_df.style.highlight_null(props="color: transparent;").format(
        "{:,.2f}", na_rep="", subset=numeric_col_names
    )
    return styled_df


def _evaluate():
    st.markdown("## 🎁 Evaluation results")
    _compute_metrics()

    if "results" in st.session_state and st.session_state["results"]:
        st.write("Below you can find the corpus results for your dataset.")

        corpus_df = _build_corpus_df()
        st.markdown("### 📊 Chart")
        _draw_corpus_scores(corpus_df)

        st.markdown("### 🗄️ Table")
        styled_df = _style_df_for_display(corpus_df)
        st.dataframe(styled_df)
        excel_link = create_download_link(corpus_df, "mateo.xlsx", "Excel file")
        txt_link = create_download_link(
            corpus_df.to_csv(index=False, encoding="utf-8", sep="\t"), "mateo.tsv", "tab-separated file"
        )
        st.markdown(f"You can download the table as an {excel_link}, or as a {txt_link}.", unsafe_allow_html=True)

        st.markdown("### 📄 LaTeX")
        latex_col_format = "l" + ("r" * len(st.session_state["metrics"]))
        st.code(
            styled_df.to_latex(
                column_format=latex_col_format,
                caption=f"Metric scores ({', '.join(list(st.session_state['metrics']))}) for"
                f" {len(st.session_state['results'])} system(s), calculated with MATEO.",
            ),
            language="latex",
        )


def main():
    _init()
    _metric_selection()
    _data_input()
    can_continue, msg = _validate_state()

    msg_container = st.empty()
    if not can_continue:
        msg_container.warning(msg)
        st.session_state["metrics"] = dict()
    else:
        msg_container.empty()
        _add_metrics_selection_to_state()

        if st.button("Evaluate MT"):
            _evaluate()
        else:
            st.write("Click the button above to start calculating the automatic evaluation scores for your data")


if __name__ == "__main__":
    main()
    # Call this to immediately disable CUDA if needed
    cli_args()
