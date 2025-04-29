import json
import os
import sys
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler
from sacrebleu.metrics.base import Metric as SbMetric
from sacrebleu.significance import PairedTest, Result, _compute_p_value, estimate_ci
from sacrebleu.utils import print_results_table

from mateo_st.metrics.metrics_constants import METRICS_META
from mateo_st.metrics.utils import build_signature


def paired_bs(
    metric_sentence_scores: dict[str, list[list[float]]],
    paired_bs_n: int = 1000,
):
    """
    :param metric_sentence_scores: a dictionary of metric_name to a list of list of sentence-level scores
    where each item corresponds to the results of one system
    :param paired_bs_n: how many partitions to use in bootstrap sampling
    :return: a dictionary with keys metrics and as values a list of dicts, where each dict
    contains the results for a system
    """
    # This seed is also used in sacrebleu
    seed = int(os.environ.get("SACREBLEU_SEED", "12345"))
    rng = np.random.default_rng(seed)
    dataset_size = len(next(iter(metric_sentence_scores.values()))[0])
    idxs = rng.choice(dataset_size, size=(paired_bs_n, dataset_size), replace=True)

    results = defaultdict(list)
    for metric_name, all_sys_scores in metric_sentence_scores.items():
        all_sys_scores = np.array(all_sys_scores)
        scores_bl, all_sys_scores = all_sys_scores[0], all_sys_scores[1:]
        # Baseline
        real_mean_bl = scores_bl.mean().item()
        # First create (n_samples, dataset_size) array that contains n_samples partitions
        # of 'dataset_size' items, randomly picked.
        # Then calculate corpus scores, which here are the average over the dataset_size,
        # so we end up with an array of (n_samples,)
        bs_scores_bl = scores_bl[idxs].mean(axis=1)
        bs_bl_mean, bl_ci = estimate_ci(bs_scores_bl)

        results[metric_name].append(Result(score=real_mean_bl, p_value=None, mean=bs_bl_mean, ci=bl_ci))

        for scores_sys in all_sys_scores:
            # The real, final score for this metric (average of all sentence scores)
            real_mean_sys = scores_sys.mean().item()
            # The remainder is borrowed and slightly adapted from sacrebleu
            diff = abs(real_mean_bl - real_mean_sys)
            # size (n_samples,)
            bs_sys_scores = scores_sys[idxs].mean(axis=1)

            # 1. bs_mean_sys: the "true" mean score estimated from bootstrap resamples of the system
            # 2. sys_ci: the 95% confidence interval around the true mean score `bs_mean_sys`
            bs_mean_sys, sys_ci = estimate_ci(bs_sys_scores)
            sample_diffs = np.abs(bs_sys_scores - bs_scores_bl)
            stats = sample_diffs - sample_diffs.mean()
            p = _compute_p_value(stats, diff)
            results[metric_name].append(Result(score=real_mean_sys, p_value=p, mean=bs_mean_sys, ci=sys_ci))

    return dict(results)


def paired_bs_sacrebleu(
    named_systems: list[tuple[str, list[str]]], metrics: dict[str, SbMetric], references: list[str], args: Namespace
):
    """
    :param named_systems: A lisf of (system_name, system_hypotheses) tuples on
    which the test will be applied.
    :param metrics: A dictionary of `Metric` instances that will be computed
    for each system.
    :param references: A sequence of reference documents with document being
    defined as a sequence of reference strings. If `None`, already cached references
    will be used through each metric's internal cache.
    :return:
    """
    res = PairedTest(
        named_systems=named_systems,
        metrics=metrics,
        n_samples=args.paired_bs_n,
        references=[references],
        test_type="bs",
    )

    signatures, results = res()
    signatures = {k: v.format(args.short) for k, v in signatures.items()}

    return results, signatures


def do_bootstrap_resampling(paired_bs_n: int = 1000):
    named_systems = []
    metric_sentence_scores = defaultdict(list)
    sb_metrics = {}
    signatures = {}
    seed = int(os.environ.get("SACREBLEU_SEED", "12345"))
    for sys_idx, results in st.session_state["results"].items():
        sys_name = st.session_state["sys_files"][sys_idx]
        if st.session_state["num_sys"] > 1:
            named_systems.append(
                (f"Baseline: {sys_name}" if sys_idx == 0 else sys_name, st.session_state["sys_segments"][sys_idx])
            )
        else:
            named_systems.append((sys_name, st.session_state["sys_segments"][sys_idx]))

        for metric_name, metric_res in results.items():
            meta = METRICS_META[metric_name]
            metric_opts = st.session_state["metrics"][metric_name].get("init", {})

            # If we already pre-calculated sentence-level scores, we can just use those directly for bootstrapping
            # If not, we are probably using sacrebleu, in which case we calculate them again later for all partitions
            if metric_res["sentences"] is not None:
                signatures[metric_name] = build_signature(paired_bs_n, seed, meta.version, metric_opts)
                metric_sentence_scores[metric_name].append(metric_res["sentences"])
            else:
                sb_metrics[metric_name] = meta.sb_class(**metric_opts)

    # "System" MUST be the first item in the dictionary because `print_results_table` skips the first item
    all_results = {"System": [named_system[0] for named_system in named_systems]}

    # Sentenve-level metrics bootstrap
    if metric_sentence_scores:
        metric_sentence_scores = dict(metric_sentence_scores)
        all_results = {**all_results, **paired_bs(metric_sentence_scores, paired_bs_n=paired_bs_n)}

    # SacreBLEU bootstraps
    args = Namespace(
        format="json",
        short=False,
        paired_bs=True,
        paired_ar=False,
        paired_bs_n=paired_bs_n,
        width=1,
    )
    if sb_metrics:
        sacrebleu_results, sb_signatures = paired_bs_sacrebleu(
            named_systems, sb_metrics, st.session_state["ref_segments"], args
        )

        signatures = {**sb_signatures, **signatures}
        all_results = {
            **all_results,
            **sacrebleu_results,
        }

    # Hacky way to retrieve `print_results_table`'s stdout (print'ed) content in a variable
    # cf. https://stackoverflow.com/a/53197293/1150683
    os_stdout = sys.stdout
    result_table_print = StringIO()
    sys.stdout = result_table_print
    print_results_table(all_results, signatures, args)
    bs_data = json.loads(result_table_print.getvalue())
    sys.stdout = os_stdout

    st.session_state["bootstrap_results"] = bs_data
    st.session_state["bootstrap_signatures"] = signatures

    return bs_data


def get_bootstrap_dataframe() -> tuple[Styler, Styler, pd.DataFrame, pd.DataFrame]:
    bs_data = do_bootstrap_resampling()

    def postprocess_data(data, *, for_display: bool = True):
        data = deepcopy(data)
        for row_idx, row in enumerate(data):
            for colname in list(row.keys()):
                if colname == "system":
                    if row_idx == 0:
                        row[colname] = f"Baseline: {row[colname]}" if st.session_state["num_sys"] > 1 else row[colname]
                    continue

                content = row[colname]
                if for_display:
                    val = f"{content['score']:.1f} ({content['mean']:.1f} ± {content['ci']:.1f})"
                else:
                    val = f"{content['score']:.2f}"

                if "p_value" in content and content["p_value"] is not None:
                    if for_display:
                        val += f"\n(p = {content['p_value']:.4f})"
                    if content["p_value"] < 0.05:
                        val += "*"

                del row[colname]
                if for_display:
                    colname = f"{colname} (μ ± 95% CI)"
                row[colname] = val

        return data

    # For display
    display_data = postprocess_data(bs_data, for_display=True)
    display_df = pd.DataFrame(display_data)
    styled_display_df = display_df.style
    column_names = display_df.columns

    def color_baseline(row: pd.Series):
        return ["color: #aeae22" for _ in row]

    if st.session_state["num_sys"] > 1:
        # Color the baseline row
        styled_display_df = styled_display_df.apply(color_baseline, subset=pd.IndexSlice[0, :])

    styled_display_df = styled_display_df.set_properties(
        **{"white-space": "pre-wrap", "text-align": "center !important"}, subset=column_names[1:]
    ).set_properties(**{"text-align": "right !important"}, subset=column_names[0])

    styled_display_df = styled_display_df.highlight_null(props="color: transparent;")

    def highlight_best(col: pd.Series, higher_better: bool):
        # We cannot just use highlight_max because that would be done on the postprocessed strings
        # "*" is used to indicate significance in the latex table (not p-value between brackets)
        floats = [float(item.strip().split(" ")[0].rstrip("*")) for item in col]
        if higher_better:
            best = max(floats)
        else:
            best = min(floats)

        return ["font-weight: bold;" if f == best else "" for f in floats]

    if st.session_state["num_sys"] > 1:
        higher_better_cols = [c for c in column_names if c not in ("TER (μ ± 95% CI)", "system")]
        lower_better_cols = ["TER (μ ± 95% CI)"] if "TER (μ ± 95% CI)" in column_names else []
        styled_display_df = styled_display_df.apply(
            highlight_best, subset=higher_better_cols, axis=0, higher_better=True
        )
        styled_display_df = styled_display_df.apply(
            highlight_best, subset=lower_better_cols, axis=0, higher_better=False
        )

    # For LaTeX
    latex_data = postprocess_data(bs_data, for_display=False)
    latex_df = pd.DataFrame(latex_data)
    styled_latex_df = latex_df.style
    column_names = latex_df.columns
    styled_latex_df = styled_latex_df.set_properties(
        **{"white-space": "pre-wrap", "text-align": "center !important"}, subset=column_names[1:]
    ).set_properties(**{"text-align": "right !important"}, subset=column_names[0])

    styled_latex_df = styled_latex_df.highlight_null(props="color: transparent;")
    if st.session_state["num_sys"] > 1:
        higher_better_cols = [c for c in column_names if c not in ("TER", "system")]
        lower_better_cols = ["TER"] if "TER" in column_names else []
        styled_latex_df = styled_latex_df.apply(highlight_best, subset=higher_better_cols, axis=0, higher_better=True)
        styled_latex_df = styled_latex_df.apply(highlight_best, subset=lower_better_cols, axis=0, higher_better=False)

    styled_latex_df = styled_latex_df.hide()

    # DataFrame for users to download without CI or p-values or asterisks
    # The main idea is for users to use this type of excel/sheet for analysis
    download_wo_ci_df = deepcopy(latex_df)
    for column in download_wo_ci_df.columns[1:]:
        download_wo_ci_df[column] = download_wo_ci_df[column].str.replace("*", "").astype(float)

    # For download with CI -- for the download without CI we use graphic_df
    download_ci_df = deepcopy(display_df)
    # Remove newlines
    download_ci_df = download_ci_df.replace({r"\s+$": "", r"^\s+": ""}, regex=True).replace(r"\n", " ", regex=True)

    return styled_display_df, styled_latex_df, download_ci_df, download_wo_ci_df
