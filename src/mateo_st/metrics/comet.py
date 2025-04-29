from dataclasses import dataclass, field
from statistics import mean
from typing import Any, ClassVar

import comet
import streamlit as st

from mateo_st.metrics.base import MetricMeta, MetricOption, NeuralMetric


comet_meta = MetricMeta(
    name="COMET",
    metric_class="neural",
    full_name="Crosslingual Optimized Metric for Evaluation of Translation",
    description_html="COMET is similar to other neural approaches in that it also finetunes existing language models,"
    " in their case"
    " <a href='https://aclanthology.org/2020.acl-main.747/' title='XLM-R paper'>XLM-R</a>. What"
    " makes COMET different, however, is that it can also consider the source text as part of the"
    " input rather than only comparing a machine translation to a reference translation.</p>",
    paper_url="https://aclanthology.org/2020.emnlp-main.213/",
    implementation_html="<p><a href='https://github.com/Unbabel/COMET' title='COMET GitHub'>COMET</a></p>",
    is_default_selected=True,
    version=comet.__version__,
    corpus_score_key="mean_score",
    sentences_score_key="scores",
    options=(
        MetricOption(
            name="model_name",
            description="COMET model to use. For supported models, look [here](https://github.com/Unbabel/COMET/blob/v2.0.1/MODELS.md).",
            default="Unbabel/wmt22-comet-da",
            choices=(
                "wmt20-comet-da",
                "wmt21-comet-da",
                "wmt21-cometinho-da",
                "eamt22-cometinho-da",
                "eamt22-prune-comet-da",
                "Unbabel/wmt22-comet-da",
            ),
            demo_choices=(
                "Unbabel/wmt22-comet-da",
                "eamt22-cometinho-da",
            ),
        ),
        MetricOption(
            name="batch_size",
            description="How many sentences to process at once. The larger the batch size, the faster the scoring but the higher the memory usage. Do not change this unless you know what you are doing. If the value is set too high it will freeze your computer and potentially crash the app!",
            default=4,
            types=(int,),
            disabled="auto",
            is_init_arg=False,
        ),
    ),
    requires_source=True,
)


@dataclass
class CometMetric(NeuralMetric):
    name: ClassVar[str] = "comet"
    meta: ClassVar[MetricMeta] = comet_meta

    model_name: str = comet_meta.options[0].default
    model: comet.models.CometModel = field(default=None, init=False)

    def __post_init__(self):
        model_path = comet.download_model(self.model_name)
        self.model = comet.load_from_checkpoint(model_path)

    def compute(
        self, references: list[str], predictions: list[str], sources: list[str], batch_size: int = 8, **kwargs
    ) -> dict:
        """Predicts the score for a batch of references and hypotheses and sources.

        :param references: list of reference sentences
        :param hypotheses: list of hypothesis sentences
        :param sources: list of source sentences
        :param kwargs: additional arguments for the model prediction
        :return: score result (dictionary)
        """
        if len(references) != len(predictions) != len(sources):
            raise ValueError("The lengths of references, hypotheses, and sources must be the same.")

        data = [{"mt": pred, "ref": ref, "src": src} for pred, ref, src in zip(predictions, references, sources)]

        return self.model.predict(data, progress_bar=False, batch_size=batch_size, **kwargs)

    @classmethod
    def postprocess_result(cls, result: dict[str, Any]):
        """Post-processes the result that is retrieved from a computed metric.

        :param result: score result (dictionary)
        :return: modified score result
        """
        corpus_key = cls.meta.corpus_score_key
        sentences_key = cls.meta.sentences_score_key
        result[corpus_key] = 100 * mean(result["scores"])
        result[sentences_key] = [score * 100 for score in result["scores"]]
        return result


@st.cache_resource(show_spinner=False, max_entries=2, ttl=60 * 60 * 24 * 30)
def get_comet_metric(model_name: str) -> CometMetric:
    """Get the COMET metric instance.

    :param model_name: name of the COMET model to use
    :return: COMET metric instance
    """
    return CometMetric(model_name=model_name)
