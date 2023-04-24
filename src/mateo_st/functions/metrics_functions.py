import base64
from io import BytesIO
from typing import List, Optional

import evaluate
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
import torch
from mateo_st.utils import COLORS_PLOTLY

from mateo_st.metrics_constants import BASELINE_METRICS, METRIC_BEST_ARROW, METRICS


try:
    # Disable all GPUS for TF evaluation - BLEURT for instance
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"
except Exception:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


@st.cache_resource(show_spinner=False)
def get_baseline_metrics(metrics: List[str]):
    metric_names = [METRICS[m] for m in metrics]
    return evaluate.combine(metric_names, force_prefix=True)


@st.cache_resource(show_spinner=False)
def get_metric(metric: str, config_name: Optional[str] = None):
    return evaluate.load(METRICS[metric], config_name=config_name)


@st.cache_data(show_spinner=False)
def calculate_bleurt(predictions, references):
    metric = get_metric("bleurt", "BLEURT-20")
    return metric.compute(predictions=predictions, references=references)


@st.cache_data(show_spinner=False)
def calculate_bertscore(predictions, references, lang: str):
    metric = get_metric("bertscore")
    return metric.compute(predictions=predictions, references=references, device="cpu", lang=lang)


@st.cache_data(show_spinner=False)
def calculate_comet(sources, predictions, references):
    metric = get_metric("comet")
    # It is not clear to me whether `gpus=0` actually disables the GPU. Is it passed to the metric config?
    return metric.compute(sources=sources, predictions=predictions, references=references, gpus=0)


def post_process_scores(scores, all_results, metric_name: str, multiplier: bool = True, clamp: bool = True):
    multiplier = 100 if multiplier else 1
    for score_idx, score in enumerate(scores):
        if clamp:
            all_results[score_idx][f"{metric_name}_score"] = min([100, max([0, score * multiplier])])
        else:
            all_results[score_idx][f"{metric_name}_score"] = score * multiplier

    return all_results


def evaluate_input(st_container):
    base_metrics = [m for m in st.session_state["metrics"] if m in BASELINE_METRICS]
    other_metrics = [m for m in st.session_state["metrics"] if m not in BASELINE_METRICS]
    torch.use_deterministic_algorithms(False)  # Disable because of CUDA error with COMET
    predictions = [st.session_state["mt_text"]] + st.session_state["other_hyps"]
    num_preds = len(predictions)

    increment = 100 // (num_preds * len(st.session_state["metrics"]))
    progress = 0
    pbar = st_container.progress(progress)

    def update_pbar(value: int):
        nonlocal progress
        progress += value
        pbar.progress(progress)

    results = []
    if st.session_state["mt_text"]:
        results.append({"translation": "MT", "text": st.session_state["mt_text"]})

    for hyp_idx, hyp in enumerate(st.session_state["other_hyps"], 1):
        results.append({"translation": f"Translation #{hyp_idx}", "text": hyp})

    fig_ct = st_container.empty()
    df_ct = st_container.empty()
    download_btn_tsv_ct = st_container.empty()

    if base_metrics:
        base_metrics = get_baseline_metrics(base_metrics)
        for pred_idx, pred in enumerate(predictions):
            res = base_metrics.compute(predictions=[pred], references=[st.session_state["ref_text"]])
            results[pred_idx].update(res)
            update_pbar(increment * len(base_metrics.evaluation_module_names))

        draw(results, fig_ct, df_ct, download_btn_tsv_ct)

    for metric in other_metrics:
        if metric == "bertscore":
            bertscore_results = calculate_bertscore(
                predictions,
                [st.session_state["ref_text"]] * num_preds,
                lang=st.session_state["tgt_lang_key"],
            )
            results = post_process_scores(bertscore_results["f1"], results, "bertscore")
        elif metric == "bleurt":
            bleurt_results = calculate_bleurt(predictions, [st.session_state["ref_text"]] * num_preds)
            results = post_process_scores(bleurt_results["scores"], results, "bleurt")
        elif metric == "comet":
            comet_results = calculate_comet(
                [st.session_state["src_text"]] * num_preds, predictions, [st.session_state["ref_text"]] * num_preds
            )
            results = post_process_scores(comet_results["scores"], results, "comet")

        update_pbar(increment * num_preds)
        draw(results, fig_ct, df_ct, download_btn_tsv_ct)

    pbar.empty()


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Sheet1")
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    format1 = workbook.add_format({"num_format": "0.00"})
    worksheet.set_column("A:A", None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def draw(results, fig_ct, df_ct, download_btn_tsv_ct):
    df = pd.DataFrame(results)

    # Remove "_score" and "sacre" (bleu's output with sacrebleu is "sacrebleu")
    df = df.rename(mapper=lambda col: col.replace("_score", "").replace("sacre", ""), axis=1)
    df = df.reindex(["translation", "text"] + [c for c in METRICS.keys() if c in df.columns], axis=1)
    df = df.rename(mapper=lambda col: f"{col} {METRIC_BEST_ARROW[col]}" if col in METRIC_BEST_ARROW else col, axis=1)

    # Reshape DataFame for plotting
    df_melt = pd.melt(df.drop(columns="text"), id_vars="translation", var_name="metric", value_name="score")
    fig = px.bar(
        df_melt,
        x="metric",
        y="score",
        color="translation",
        barmode="group",
        color_discrete_sequence=COLORS_PLOTLY["default"],
    )

    fig_ct.plotly_chart(fig)

    # Remove arrows from cols for writing to file/display
    df = df.rename(mapper=lambda col: col.replace("↑", "").replace("↓", "").strip(), axis=1)
    # Add source/ref rows
    src_df = pd.DataFrame(
        [
            ["Source", st.session_state["src_text"]] + [pd.NA] * (len(df.columns) - 2),
            ["Ref.", st.session_state["ref_text"]] + [pd.NA] * (len(df.columns) - 2),
        ],
        columns=df.columns,
    )
    df = pd.concat([src_df, df], ignore_index=True).reset_index(drop=True)
    df = df.set_index("translation")

    # Round to two decimals for now, which should make copying into Excel easier
    # TODO: find a better solution to work with Excel as well as giving very precise data (not rounded)
    rounded_df = df.replace(pd.NA, np.nan)
    rounded_df.iloc[:, 1:] = rounded_df.iloc[:, 1:].astype(float).round(decimals=2).copy()

    # Hide NANs from display
    numeric_col_names = rounded_df.columns[1:].tolist()
    styled_df = rounded_df.style.highlight_null(props="color: transparent;").format(
        "{:,.2f}", na_rep="", subset=numeric_col_names
    )
    df_ct.dataframe(styled_df)

    # Let users download the unrounded data
    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(df.to_csv(encoding="utf-8", sep="\t").encode("utf-8").encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(df.to_csv(encoding="utf-8", sep="\t").encode("utf-8")).decode()
    download_btn_tsv_ct.markdown(
        f"""
        <a download="mateo.tsv" href="data:file/txt;base64,{b64}">Download TSV</a>
        """,
        unsafe_allow_html=True,
    )
