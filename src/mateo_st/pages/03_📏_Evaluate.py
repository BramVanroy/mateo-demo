import logging
import warnings
from pathlib import Path
from random import shuffle
from typing import Any, Optional, Type

import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sacrebleu.metrics.base import Metric as SbMetric

from mateo_st import __version__ as mateo_version
from mateo_st.metrics.base import NeuralMetric
from mateo_st.metrics.metrics_constants import (
    METRICS_META,
    NEURAL_METRIC_GETTERS,
)
from mateo_st.metrics.significance import get_bootstrap_dataframe
from mateo_st.metrics.utils import merge_batched_results
from mateo_st.utils import (
    cli_args,
    create_download_link,
    get_uploaded_file_as_strio,
    isfloat,
    isint,
    load_css,
    print_citation_info,
)


logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _init():
    st.set_page_config(page_title="Evaluate Machine Translations | MATEO", page_icon="📏", layout="wide")
    load_css("base")
    load_css("evaluate")

    if "ref_segments" not in st.session_state:
        st.session_state["ref_segments"] = []

    if "ref_file" not in st.session_state:
        st.session_state["ref_file"] = None

    if "src_segments" not in st.session_state:
        st.session_state["src_segments"] = []

    if "src_file" not in st.session_state:
        st.session_state["src_file"] = None

    if "sys_segments" not in st.session_state:
        st.session_state["sys_segments"] = {}

    if "sys_files" not in st.session_state:
        st.session_state["sys_files"] = {}

    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {}

    if "results" not in st.session_state:
        st.session_state["results"] = {}

    if "bootstrap_results" not in st.session_state:
        st.session_state["bootstrap_results"] = {}

    if "bootstrap_signatures" not in st.session_state:
        st.session_state["bootstrap_signatures"] = {}

    if "num_sys" not in st.session_state:
        st.session_state["num_sys"] = 1

    st.title("📏 Evaluate")
    st.markdown(
        "Here you can evaluate machine translation systems on your own test set of reference translations."
        " First specify the metrics and metric options to use, and then upload your data."
    )

    print_citation_info()


def _metric_selection():
    st.markdown("## ✨ Metric selection")

    if cli_args().demo_mode:
        st.info(
            "Some advanced (non-default) options for the neural metrics are not available on this web"
            " page but can be activated easily when you run the demo"
            " [on your own device or server](https://github.com/BramVanroy/mateo-demo/#self-hosting).",
            icon="ℹ️",
        )

    metric_inp_col_left, metric_inp_col_right = st.columns(2)

    # Iterate over all the metrics in METRIC_META and add each one and all its options
    for metric_idx, (ugly_metric_name, meta) in enumerate(METRICS_META.items()):
        metric_container = metric_inp_col_left if metric_idx % 2 == 0 else metric_inp_col_right
        pretty_metric_name = meta.name
        metric_container.checkbox(f"Use {pretty_metric_name}", key=ugly_metric_name, value=meta.is_default_selected)

        # Add options as an "expander"
        if meta.options:
            expander = metric_container.expander(f"{meta.name} options")
            for opt in meta.options:
                opt_name = opt.name
                has_choices = bool(opt.choices)

                opt_label = f"{ugly_metric_name}--{opt_name}"

                # In demo mode, automatically disable some option fields
                if opt.disabled == "auto":
                    opt.disabled = cli_args().demo_mode

                kwargs = {
                    "label": f"{opt_name} (default: '{opt.default}')",
                    "help": opt.description,
                    "key": opt_label,
                    "disabled": opt.disabled,
                }

                # If this option is one with choices or with free input
                if has_choices:
                    if cli_args().demo_mode:
                        expander.selectbox(
                            options=opt.demo_choices, index=opt.demo_choices.index(opt.default), **kwargs
                        )
                    else:
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
                            expander.number_input(**kwargs, step=0.01 if float in opt.types else 1, min_value=0)
                    elif dtype is bool:
                        expander.checkbox(**kwargs)


def _data_input():
    st.markdown("## 📄 Input data")
    st.write(
        "Add a reference file and one or more files with translations. One line per file."
        " Cannot contain empty lines and must be in UTF8!"
    )

    def read_file(uploaded_file) -> list[str]:
        if uploaded_file is not None:
            if (stringio := get_uploaded_file_as_strio(uploaded_file)) is not None:
                return stringio.read().splitlines()

        return []

    ref_inp_col, src_inp_col = st.columns(2)

    ref_file = ref_inp_col.file_uploader("Reference file")
    st.session_state["ref_segments"] = read_file(ref_file)
    st.session_state["ref_file"] = ref_file.name if ref_file else None

    # Check whether any of the selected metrics require source input
    # If so, use a two-col layout for the input buttons, if not just use full-width reference input
    requires_source = any(
        meta.requires_source and name in st.session_state and st.session_state[name]
        for name, meta in METRICS_META.items()
    )
    # Always give the option to upload a source file (can be useful for the segment-level viz/table)
    src_file = src_inp_col.file_uploader("Source file" if requires_source else "Source file (optional)")
    st.session_state["src_segments"] = read_file(src_file)
    st.session_state["src_file"] = src_file.name if src_file else None

    max_sys_inp_col, _ = st.columns(2)
    max_sys = cli_args().eval_max_sys
    max_sys_inp_col.number_input(
        f"How many systems do you wish to compare? (max. {max_sys})",
        step=1,
        min_value=1,
        max_value=max_sys,
        key="num_sys",
    )

    # Iterate over i..max_value. Reason is that we need to delete sys_idx if it does not exist anymore
    # This can happen when a user first has three systems and then changes it back to 1
    sys_inp_col_left, sys_inp_col_right = st.columns(2)
    for sys_idx in range(1, max_sys + 1):
        if sys_idx <= st.session_state["num_sys"]:
            sys_container = sys_inp_col_left if sys_idx % 2 != 0 else sys_inp_col_right
            if st.session_state["num_sys"] > 1 and sys_idx == 1:
                sys_file = sys_container.file_uploader(f"System #{sys_idx} (serves as baseline)")
            else:
                sys_file = sys_container.file_uploader(f"System #{sys_idx} file")
            st.session_state["sys_segments"][sys_idx] = read_file(sys_file)
            st.session_state["sys_files"][sys_idx] = Path(sys_file.name).stem if sys_file else None
        else:
            if sys_idx in st.session_state["sys_segments"]:
                del st.session_state["sys_segments"][sys_idx]

            if sys_idx in st.session_state["sys_files"]:
                del st.session_state["sys_files"][sys_idx]


def _validate_state() -> tuple[bool, str]:
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
    st.session_state["metrics"] = {}
    for metric_name, meta in METRICS_META.items():
        if metric_name in st.session_state and st.session_state[metric_name]:
            st.session_state["metrics"][metric_name] = {"init": {}, "call": {}}
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

                if opt.is_init_arg:
                    st.session_state["metrics"][metric_name]["init"][opt_name] = opt_val
                else:
                    st.session_state["metrics"][metric_name]["call"][opt_name] = opt_val


def _load_metric(metric_name: str, **options) -> NeuralMetric:
    """Load an individual neural metric. This is a short utility function so that
    each metric can have its own initialization function which can also be cached.
    Caching is done at the individual level for more control if needed.

    :param metric_name: metric name
    :param options: extra options, in fact the init arguments
    :return: loaded metric
    """
    return NEURAL_METRIC_GETTERS[metric_name](**options)


@st.cache_resource(show_spinner=False, max_entries=24)
def _load_sacrebleu_metric(sb_class: Type[SbMetric], **options) -> SbMetric:
    """Load an individual metric with SacreBLEU

    :param sb_class: a sacrebleu Class to instantiate this metric with
    :param options: initialization options
    :return: loaded sacrebleu metric
    """
    return sb_class(**options)


def batchify(predictions: list[str], references: list[str], sources: Optional[list[str]] = None, batch_size: int = 16):
    samples = list(zip(predictions, references, sources if sources else ([""] * len(references))))

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        preds, refs, srcs = list(zip(*batch))
        batch = {"predictions": preds, "references": refs}
        if sources:
            batch["sources"] = srcs
        yield batch


@st.cache_data(show_spinner=False, ttl=21600, max_entries=1024)
def _compute_metric(
    metric_name: str,
    init_kwargs: dict[str, Any] = {},
    call_kwargs: dict[str, Any] = {},
    *,
    predictions: list[str],
    references: list[str],
    sources: Optional[list[str]] = None,
    sb_class: Optional[Type[SbMetric]] = None,
    _sys_idx: Optional[int] = None,
):
    pbar = st.empty()
    try:
        if sb_class is None:
            metric = _load_metric(metric_name, **init_kwargs)
            progress_batch_size = call_kwargs.get("batch_size", init_kwargs.get("batch_size", 16))
            msg = f"Calculating `{METRICS_META[metric_name].name}`{' for system #' + str(_sys_idx) if _sys_idx is not None else ''}"
            pbar.progress(0)
            num_batches = max(1, len(references) // progress_batch_size)
            increment = max(1, 100 // num_batches)
            progress = 0
            results = []
            for batch in batchify(predictions, references, sources, batch_size=progress_batch_size):
                if sources and "sources" in batch:
                    results.append(
                        metric.compute(
                            predictions=batch["predictions"],
                            references=batch["references"],
                            sources=batch["sources"],
                            **call_kwargs,
                        )
                    )
                else:
                    results.append(metric.compute(predictions=batch["predictions"], references=batch["references"]))
                progress += increment
                pbar.progress(min(progress, 100), text=f"{msg}: {min(progress, 100):.2f}%")

            pbar.empty()
            result = merge_batched_results(results)
            result = metric.postprocess_result(result)
        else:
            metric = _load_sacrebleu_metric(sb_class, **init_kwargs)
            result = metric.corpus_score(hypotheses=predictions, references=[references], **call_kwargs)
            # Sacrebleu returns special Score classes -- convert to dict
            result = vars(result)
    except Exception as exc:
        pbar.empty()
        raise exc
    else:
        return result


def _compute_metrics():
    results = {}
    pbar_text_ct = st.empty()
    pbar = st.progress(0)
    increment = max(1, 100 // (len(st.session_state["sys_segments"]) * len(st.session_state["metrics"])))
    progress = 0

    error_ct = st.empty()
    got_exception = False

    for sys_idx, sys_segs in st.session_state["sys_segments"].items():
        if got_exception:
            break
        results[sys_idx] = {}
        # Shuffle metrics in hopes of not simultaneously using the same metric, especially in class room settings
        # where similar jobs will be executed at the same time. We don't want to overload the server
        # when everyone is suddenly calculating BLEURT at the same time. So if we can shuffle the order of the metrics
        # some of them will first calculate bertscore or BLEU or BLEURT, etc.
        shuffled_metrics = list(st.session_state["metrics"].items())
        shuffle(shuffled_metrics)

        for metric_name, init_and_call_opts in shuffled_metrics:
            init_kwargs = init_and_call_opts.get("init", {})
            call_kwargs = init_and_call_opts.get("call", {})

            meta = METRICS_META[metric_name]

            if sys_idx == 1:
                msg = f"(Down)loading metric <code>{meta.name}</code> and evaluating system #{sys_idx}."
                if meta.metric_class == "neural":
                    msg += (
                        f"<br><code>{meta.name}</code> is a neural metric, so downloading and calculating may take"
                        f" a long while, depending on the size of your dataset."
                    )
            else:
                msg = f"Evaluating system #{sys_idx} with <code>{meta.name}</code>"
                if meta.metric_class == "neural":
                    msg += (
                        f"<br><code>{meta.name}</code> is a neural metric, so calculating may take"
                        f" a long while, depending on the size of your dataset."
                    )

            pbar_text_ct.markdown(f'<p style="font-size: 0.8em">{msg}</code></p>', unsafe_allow_html=True)

            try:
                result = _compute_metric(
                    metric_name,
                    init_kwargs=init_kwargs,
                    call_kwargs=call_kwargs,
                    predictions=sys_segs,
                    references=st.session_state["ref_segments"],
                    sources=st.session_state["src_segments"] if meta.requires_source else None,
                    sb_class=meta.sb_class,
                    _sys_idx=sys_idx,
                )
            except Exception as exc:
                error_ct.exception(exc)
                pbar_text_ct.empty()
                pbar.empty()
                got_exception = True
                st.session_state["results"] = {}
                break
            else:
                error_ct.empty()
                results[sys_idx][metric_name] = {
                    "corpus": result[meta.corpus_score_key],
                    "sentences": result[meta.sentences_score_key] if meta.sentences_score_key else None,
                }

                progress += increment
                pbar.progress(min(progress, 100))

    if not got_exception:
        pbar_text_ct.empty()
        pbar.empty()
        # Sort the results by metric name to have consistent ordering (they were shuffled before)
        for sys_idx, sys_results in results.items():
            sorted_results = dict(sorted(sys_results.items(), key=lambda item: item[0]))
            results[sys_idx] = sorted_results
        st.session_state["results"] = results


def _build_corpus_df():
    data = []
    for sys_idx, results in st.session_state["results"].items():
        sys_data = {"system": st.session_state["sys_files"][sys_idx]}
        for metric_name, metric_res in results.items():
            sys_data[metric_name] = metric_res["corpus"]
        data.append(sys_data)

    df = pd.DataFrame(data)

    # Remove "sacre" (bleu's output with sacrebleu is "sacrebleu")
    def col_mapper(colname: str):
        if colname in METRICS_META:
            return f"{colname} {'↑' if METRICS_META[colname].higher_better else '↓'}"
        else:
            return colname

    df = df.rename(mapper=col_mapper, axis=1)
    df = df.rename(mapper=lambda col: col.replace("sacre", ""), axis=1)
    return df


def _build_sentence_df(include_sys_translations: bool = True):
    data = []
    has_src = "src_segments" in st.session_state and st.session_state["src_segments"]

    for metric_name in st.session_state["metrics"].keys():
        if not METRICS_META[metric_name].segment_level:
            continue

        for item_idx in range(len(st.session_state["ref_segments"])):
            item = {
                "metric": metric_name.replace("sacre", ""),
            }

            if has_src:
                # If a user uploaded a source file, include the source column even if the metric
                # is not a source-based metric
                item["src"] = st.session_state["src_segments"][item_idx]

            item["ref"] = st.session_state["ref_segments"][item_idx]
            for sys_idx, results in st.session_state["results"].items():
                sys_name = st.session_state["sys_files"][sys_idx]
                item[sys_name] = (
                    st.session_state["sys_segments"][sys_idx][item_idx] if include_sys_translations else None
                )
                item[f"{sys_name}_score"] = results[metric_name]["sentences"][item_idx]

            data.append(item)

    df = pd.DataFrame(data)
    # Drop empty columns, which will happen when include_sys_translations is None
    df = df.dropna(axis=1, how="all")

    return df


def _draw_corpus_scores(df):
    # Barplot
    # Reshape DataFame for plotting
    df_melt = pd.melt(df, id_vars="system", var_name="metric", value_name="score")

    bar_plot_tab, radar_plot_tab = st.tabs(["Bar plot", "Radar plot"])
    bar_fig = px.bar(
        df_melt,
        x="metric",
        y="score",
        color="system" if st.session_state["num_sys"] > 1 else None,
        barmode="group",
        template="plotly",
    )

    bar_plot_tab.plotly_chart(bar_fig)

    # Radar plot
    systems = df["system"].unique().tolist()
    metrics = [col for col in df.columns if col != "system"]
    radar_fig = go.Figure()

    for system in systems:
        radar_fig.add_trace(
            go.Scatterpolar(
                r=df.loc[df["system"] == system].values.flatten().tolist()[1:],
                theta=metrics,
                fill="toself",
                name=system,
            )
        )

    radar_fig.update_layout(showlegend=True, template="plotly")
    radar_plot_tab.plotly_chart(radar_fig)


def _segment_level_comparison_viz(sentence_df: pd.DataFrame):
    st.markdown(
        "📊 **Figure**: Here you can get a glance of how the system(s) perform on a per-sample level. If you are"
        " evaluating multiple systems, this can be useful to find samples for which two systems perform similarly or very differently."
    )

    if len(st.session_state["ref_segments"]) > 500:
        st.warning(
            f"Your dataset is relatively large ({len(st.session_state['ref_segments']):,d}) so this figure"
            f" may not be as useful and it may be slow to navigate.",
            icon="ℹ️",
        )

    metric_names = sentence_df["metric"].unique().tolist()
    # dict() constructor, as suggested by ruff, incompatible with DataFrameGroupBy
    grouped_df = {metric_name: df for metric_name, df in sentence_df.groupby("metric")}  # noqa: C416
    pretty_names = [METRICS_META[metric_name].name for metric_name in metric_names]

    for metric_name, tab in zip(metric_names, st.tabs(pretty_names)):
        metricdf = grouped_df[metric_name]
        metricdf["sample"] = range(1, len(metricdf.index) + 1)
        metricdf = metricdf.drop(columns="metric").reset_index(drop=True)

        def normalize_score_col(colname: str) -> str:
            # Have to get rid of dots in filenames
            # https://github.com/altair-viz/altair/issues/990
            return colname.replace("_score", "").replace(".", "-")

        has_src = "src_segments" in st.session_state and st.session_state["src_segments"]

        if has_src:
            # If a user uploaded a source file, include the source column even if the metric
            # is not a source-based metric
            metricdf["src"] = st.session_state["src_segments"].copy()

        # Rename columns so that score columns have no special ending, and columns with translations end in _text
        sys_score_cols = [c for c in metricdf.columns if c.endswith("_score")]

        metricdf = metricdf.rename(
            columns={c.replace("_score", ""): f"{normalize_score_col(c)}_text" for c in sys_score_cols}
        )
        metricdf = metricdf.rename(columns={c: normalize_score_col(c) for c in sys_score_cols})

        sys_score_cols = [normalize_score_col(c) for c in sys_score_cols]
        sys_text_names = [f"{c}_text" for c in sys_score_cols]

        id_vars = (["sample", "src", "ref"] if has_src else ["sample", "ref"]) + sys_text_names
        df_melt = metricdf.melt(id_vars=id_vars, value_vars=sys_score_cols, var_name="system", value_name="score")
        nearest_sample = alt.selection_point(
            nearest=True, on="mouseover", fields=["sample"], empty=False, clear="mouseout"
        )
        chart = (
            alt.Chart(df_melt)
            .mark_circle()
            .encode(
                x=alt.X("sample:Q", title="Sample", axis=alt.Axis(labels=False)),
                y=alt.Y("score:Q", title="Score"),
                color="system:N",
                tooltip=id_vars + ["system", "score"],
            )
            .add_params(nearest_sample)
            .interactive()
        )

        # Draw a vertical rule at the location of the selection
        vertical_line = (
            alt.Chart(df_melt)
            .mark_rule(color="gray")
            .encode(
                x="sample:Q",
                opacity=alt.condition(nearest_sample, alt.value(1.0), alt.value(0.0)),
            )
            .transform_filter(nearest_sample)
        )

        # Combine the chart and vertical_line
        layer = alt.layer(chart, vertical_line)

        tab.altair_chart(layer, use_container_width=True)


def _evaluate():
    st.markdown("## 🎁 Evaluation results (corpus)")
    _compute_metrics()

    if "results" in st.session_state and st.session_state["results"]:
        st.markdown(
            "📊 **Figures**: You can download figures by hovering over them and clicking the"
            " camera icon in the top right."
        )
        # First build a df solely based on corpus results -- without bootstrap resampling
        # We do this so the user already has something to look at while bootstrap is being done in the background
        corpus_df = _build_corpus_df()
        _draw_corpus_scores(corpus_df)

        # We need the resampled results to show the table
        bs_info = st.info("Bootstrap resampling...")
        # 1: to display as table; 2: to .to_latex and show as code;
        # 3. to download (with CI); 4. to download (without CI)
        styled_display_df, styled_latex_df, download_ci_df, download_wo_ci_df = get_bootstrap_dataframe()
        if "bootstrap_results" in st.session_state and st.session_state["bootstrap_results"]:
            st.markdown(
                "🗄️ **Table**: this table includes corpus-level results as well as the mean and 95% confidence"
                " intervals between brackets that have been calculated with (paired) bootstrap resampling"
                " (n=1000), compatible with the implementation in"
                " [SacreBLEU](https://github.com/mjpost/sacrebleu/blob/38256a74f15d35d07f24976f709edefe7a027f0b/sacrebleu/significance.py#L199)."
            )

            if st.session_state["num_sys"] > 1:
                st.markdown(
                    "The p-values indicate the significance of the difference between a system and the baseline."
                    " An asterisk * indicates that a system differs significantly from the baseline (p<0.05). The"
                    " best system is highlighted in **bold**."
                )

            bs_info.empty()
            st.table(styled_display_df)

            # Download tables
            excel_link_with_ci = create_download_link(download_ci_df, "mateo-evaluation-ci.xlsx", "Excel file")
            txt_link_with_ci = create_download_link(
                download_ci_df.to_csv(index=False, encoding="utf-8", sep="\t"),
                "mateo-evaluation-ci.tsv",
                "tab-separated file",
            )
            excel_sentences = create_download_link(download_wo_ci_df, "mateo-evaluation.xlsx", "Excel file")
            txt_link_wo_ci = create_download_link(
                download_wo_ci_df.to_csv(index=False, encoding="utf-8", sep="\t"),
                "mateo-evaluation.tsv",
                "tab-separated file",
            )
            st.markdown(
                f"📥 **Download** the table with or without confidence intervals and full p-values:\n"
                f"- with: {excel_link_with_ci} / {txt_link_with_ci}\n"
                f"- without: {excel_sentences} / {txt_link_wo_ci}",
                unsafe_allow_html=True,
            )

            # Signatures
            if "bootstrap_signatures" in st.session_state and st.session_state["bootstrap_signatures"]:
                st.markdown(
                    "💡 **Signatures**: it's a good idea to report these in your paper so others know exactly"
                    " which configuration you used!"
                )
                signatures = [
                    f"{metric}: {sig}|mateo:{mateo_version}"
                    for metric, sig in st.session_state["bootstrap_signatures"].items()
                ]
                st.code("\n".join(signatures))

            # Latex
            st.markdown(
                "📝 **LaTeX**: Make sure to include `booktabs` at the top of your LaTeX file: `\\usepackage{booktabs}`"
            )
            latex_caption = "Evaluation results generated with MATEO."
            if st.session_state["num_sys"] > 1:
                latex_caption += " * indicates a significant difference with the first row (baseline)."

            st.code(
                styled_latex_df.to_latex(
                    caption=latex_caption,
                    convert_css=True,
                    hrules=True,
                ),
                language="latex",
            )

    # Sentence
    st.markdown("## 🎁 Evaluation results (sentences)")
    with st.expander("About sentence-level scores"):
        st.write(
            "Some metrics have a corresponding sentence-level score, which you can then download here. For"
            " instance, the COMET corpus score is simply the average (arithmetic mean) of all the sentence"
            " scores. This is not the case for all metrics. Corpus BLEU for instance, is not simply the"
            " arithmetic mean of all sentence-level BLEU scores but is calculated by its geometric mean and"
            " modified by a brevity penalty. As such, sentence-level BLEU scores, or similar, are not given"
            " because they individually do not give a good representation of a system's performance."
        )

    if "results" in st.session_state and st.session_state["results"]:
        sentence_df = _build_sentence_df(include_sys_translations=True)
        if not sentence_df.empty:
            metric_names = sentence_df["metric"].unique().tolist()
            if metric_names:
                # Visualization segment-level
                if len(st.session_state["ref_segments"]) <= 1000:
                    _segment_level_comparison_viz(sentence_df)
                else:
                    st.warning(
                        f"Your dataset is relatively large ({len(st.session_state['ref_segments']):,d}) so"
                        f" the figure to visualize sentence-level performances is disabled (max.: 1000). It"
                        f" would be slow and hard to navigate. Instead you can manually analyse the data in the"
                        f" table below.",
                        icon="ℹ️",
                    )

                # Table segment-level
                st.markdown(
                    "🗄️ **Table**: In this large table you have access to all sentence-level scores for appropriate"
                    " metrics. You can use this data for a fine-grained or qualitative analysis on a per-sample basis."
                )
                grouped_df = {metric_name: df for metric_name, df in sentence_df.groupby("metric")}  # noqa: C416
                pretty_names = [METRICS_META[metric_name].name for metric_name in metric_names]

                for metric_name, tab in zip(metric_names, st.tabs(pretty_names)):
                    metricdf = grouped_df[metric_name]
                    metricdf = metricdf.drop(columns="metric").reset_index(drop=True)
                    tab.dataframe(metricdf)

                excel_sentences_dl = create_download_link(
                    sentence_df, "mateo-sentences.xlsx", "Excel file", df_groupby="metric"
                )
                st.markdown(
                    f"You can download the table as an {excel_sentences_dl}. Metrics are separated in sheets.",
                    unsafe_allow_html=True,
                )
        else:
            segment_level_metrics = [meta.name for m, meta in METRICS_META.items() if meta.segment_level]
            st.info(
                f"No segment-level metrics calculated. Segment level metrics are: {', '.join(segment_level_metrics)}."
            )


def main():
    _init()
    _metric_selection()
    _data_input()
    can_continue, msg = _validate_state()

    msg_container = st.empty()
    if not can_continue:
        msg_container.warning(msg)
        st.session_state["metrics"] = {}
    else:
        msg_container.empty()

        if st.button("Evaluate MT"):
            _add_metrics_selection_to_state()
            _evaluate()
        else:
            st.write("Click the button above to start calculating the automatic evaluation scores for your data")


if __name__ == "__main__":
    main()
    # Call this to immediately disable CUDA if needed
    cli_args()
