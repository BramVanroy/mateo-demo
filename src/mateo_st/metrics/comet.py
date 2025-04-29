from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict

import comet
from mateo_st.metrics.base import MetricMeta, MetricOption, NeuralMetric


@dataclass
class CometMeta(MetricMeta):
    def postprocess_result(self, result: Dict[str, Any]):
        """Post-processes the result that is retrieved from a computed metric.

        :param result: score result (dictionary)
        :return: modified score result
        """
        corpus_key = self.corpus_score_key
        sentences_key = self.sentences_score_key
        result[corpus_key] = 100 * mean(result["scores"])
        result[sentences_key] = [score * 100 for score in result["scores"]]
        return result


comet_meta = CometMeta(
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
            name="config_name",
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
    ),
    requires_source=True,
)


@dataclass
class CometMetric(NeuralMetric):
    name = "comet"
    meta = comet_meta

    model_name: str = comet_meta.options[0].default
    model: comet.models.CometModel = field(default=None, init=False)

    def __post_init__(self):
        model_path = comet.download_model(self.model_name)
        self.model = comet.load_from_checkpoint(model_path)

    def compute(self, references: list[str], predictions: list[str], sources: list[str], batch_size: int = 1, **kwargs) -> Any:
        """Predicts the score for a batch of references and hypotheses and sources.

        :param references: list of reference sentences
        :param hypotheses: list of hypothesis sentences
        :param sources: list of source sentences
        :param kwargs: additional arguments for the model prediction
        :return: score result (dictionary)
        """
        print(f"Using {self.model_name} model for COMET scoring.")
        if len(references) != len(predictions) != len(sources):
            raise ValueError("The lengths of references, hypotheses, and sources must be the same.")

        data = [{"mt": pred, "ref": ref, "src": src} for pred, ref, src in zip(predictions, references, sources)]

        return self.model.predict(data, progress_bar=False, batch_size=batch_size, **kwargs)
