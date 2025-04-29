from collections import defaultdict
from typing import Any, Dict, List

from mateo_st.metrics.bertscore import BertScoreMetric, bertscore_meta
from mateo_st.metrics.bleu import bleu_meta
from mateo_st.metrics.bleurt import BleurtMetric, bleurt_meta
from mateo_st.metrics.chrf import chrf_meta
from mateo_st.metrics.comet import CometMetric, comet_meta
from mateo_st.metrics.ter import ter_meta


METRICS_META = {
    "bertscore": bertscore_meta,
    "bleu": bleu_meta,
    "bleurt": bleurt_meta,
    "chrf": chrf_meta,
    "comet": comet_meta,
    "ter": ter_meta,
}

NEURAL_METRICS = {"comet": CometMetric, "bleurt": BleurtMetric, "bertscore": BertScoreMetric}

DEFAULT_METRIC_NAMES = {name for name, meta in METRICS_META.items() if meta.is_default_selected}
BASELINE_METRIC_NAMES = {name for name, meta in METRICS_META.items() if meta.metric_class == "baseline"}


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
