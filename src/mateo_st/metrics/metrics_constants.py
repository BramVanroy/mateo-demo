from mateo_st.metrics.bertscore import BertScoreMetric, bertscore_meta, get_bertscore_metric
from mateo_st.metrics.bleu import bleu_meta
from mateo_st.metrics.bleurt import BleurtMetric, bleurt_meta, get_bleurt_metric
from mateo_st.metrics.chrf import chrf_meta
from mateo_st.metrics.comet import CometMetric, comet_meta, get_comet_metric
from mateo_st.metrics.ter import ter_meta


METRICS_META = {
    "bertscore": bertscore_meta,
    "bleu": bleu_meta,
    "bleurt": bleurt_meta,
    "chrf": chrf_meta,
    "comet": comet_meta,
    "ter": ter_meta,
}

NEURAL_METRIC_GETTERS = {
    "comet": get_comet_metric,
    "bleurt": get_bleurt_metric,
    "bertscore": get_bertscore_metric,
}

NEURAL_METRICS = {"comet": CometMetric, "bleurt": BleurtMetric, "bertscore": BertScoreMetric}
DEFAULT_METRIC_NAMES = {name for name, meta in METRICS_META.items() if meta.is_default_selected}
BASELINE_METRIC_NAMES = {name for name, meta in METRICS_META.items() if meta.metric_class == "baseline"}
