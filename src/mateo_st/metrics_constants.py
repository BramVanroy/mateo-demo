from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List

from mateo_st.metrics.bertscore import bertscore_meta
from mateo_st.metrics.bleu import bleu_meta
from mateo_st.metrics.bleurt import bleurt_meta
from mateo_st.metrics.chrf import chrf_meta
from mateo_st.metrics.comet import comet_meta
from mateo_st.metrics.ter import ter_meta


METRICS_META = {
    "bertscore": bertscore_meta,
    "bleu": bleu_meta,
    "bleurt": bleurt_meta,
    "chrf": chrf_meta,
    "comet": comet_meta,
    "ter": ter_meta,
}

DEFAULT_METRICS = {name for name, meta in METRICS_META.items() if meta.is_default_selected}
BASELINE_METRICS = {name for name, meta in METRICS_META.items() if meta.metric_class == "baseline"}


def merge_batched_results(metric_name: str, results: List[Dict[str, Any]]):
    if metric_name in ("bertscore", "bleurt", "comet"):
        result = defaultdict(list)
        for batch_result in results:
            # score_key is something like "f1" or "scores"
            for score_key, scores in batch_result.items():
                if isinstance(scores, list):
                    result[score_key].extend(scores)
                else:
                    result[score_key].append(scores)
    else:
        raise ValueError(
            "Unsupported metric for batch processing. Make sure that this is a metric that simply"
            " averages the sentence-level scores, otherwise we cannot do batched predictions."
        )
    return dict(result)


def postprocess_result(metric_name: str, result: Dict[str, Any]):
    """Post-processes the result that is retrieved from Metric.compute.

    :param metric_name: the metric name
    :param result: score result (dictionary)
    :return: modified score result
    """
    corpus_key = METRICS_META[metric_name].corpus_score_key
    sentences_key = METRICS_META[metric_name].sentences_score_key
    if metric_name == "bertscore":
        result[corpus_key] = 100 * mean(result["f1"])
        result[sentences_key] = [score * 100 for score in result["f1"]]
    elif metric_name == "bleurt":
        result[corpus_key] = 100 * mean(result["scores"])
        result[sentences_key] = [score * 100 for score in result["scores"]]
    elif metric_name == "comet":
        result[corpus_key] = 100 * mean(result["scores"])
        result[sentences_key] = [score * 100 for score in result["scores"]]

    return result
