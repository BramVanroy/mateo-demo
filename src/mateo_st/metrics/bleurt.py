import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, ClassVar

import streamlit as st
from bleurt import score as bleurt_score
from huggingface_hub import snapshot_download

from mateo_st.metrics.base import MetricMeta, MetricOption, NeuralMetric


bleurt_meta = MetricMeta(
    name="BLEURT",
    metric_class="neural",
    full_name="Bilingual Evaluation Understudy with Representations from Transformers",
    description_html="BLEURT utilizes large language models <a href='https://aclanthology.org/N19-1423/'"
    " title='BERT paper'>BERT</a> and"
    " <a href='https://openreview.net/forum?id=xpFFI_NtgpW' title='RemBERT paper'"
    ">RemBERT</a> to compare a machine translation to a reference translation."
    " In their work, they highlight that most of their improvements are thanks to using synthetic"
    " data and pretraining on many different tasks such as back translation, entailment, and"
    " predicting existing MT metrics such as BLEU and BERTScore.</p>",
    paper_url="https://aclanthology.org/2020.acl-main.704/",
    implementation_html="<p><a href='https://github.com/google-research/bleurt' title='BLEURT GitHub'>BLEURT</a></p>",
    version="commit cebe7e6",
    corpus_score_key="mean_score",  # Manually added in postprocessing
    sentences_score_key="scores",
    options=(
        MetricOption(
            name="model_name",
            description="BLEURT trained model to use. See"
            " [this overview](https://github.com/google-research/bleurt/blob/master/checkpoints.md#distilled-models)"
            " for more information about the models. Note that the default model is very slow and you may"
            " wish to prefer one of the others!",
            default="BLEURT-20",
            choices=(
                "bleurt-tiny-128",
                "bleurt-tiny-512",
                "bleurt-base-128",
                "bleurt-base-512",
                "bleurt-large-128",
                "bleurt-large-512",
                "BLEURT-20-D3",
                "BLEURT-20-D6",
                "BLEURT-20-D12",
                "BLEURT-20",
            ),
            demo_choices=(
                "bleurt-base-128",
                "bleurt-base-512",
                "BLEURT-20",
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
)


@dataclass
class BleurtMetric(NeuralMetric):
    name: ClassVar[str] = "bleurt"
    meta: ClassVar[MetricMeta] = bleurt_meta

    model_name: str = bleurt_meta.options[0].default
    model: bleurt_score.BleurtScorer = field(default=None, init=False)

    def __post_init__(self):
        # Even disabling symlinks does not work (at least on Windows), so we need to set the local dir
        # to avoid symlink issues. The .sym files cannot be read by BLEURT.
        local_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / self.model_name
        if not local_dir.exists():
            local_dir.mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(repo_id=f"BramVanroy/{self.model_name}", local_dir=local_dir)
        self.model = bleurt_score.BleurtScorer(model_path)

    def compute(self, references: list[str], predictions: list[str], batch_size: int = 4) -> dict:
        """Predicts the score for a batch of references and hypotheses.

        :param references: list of reference sentences
        :param hypotheses: list of hypothesis sentences
        :param batch_size: batch size for processing
        :return: score result (dictionary)
        """
        if len(references) != len(predictions):
            raise ValueError("The lengths of references and hypotheses must be the same.")

        scores = self.model.score(
            references=references,
            candidates=predictions,
            batch_size=batch_size,
        )

        return {"scores": scores}

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
def get_bleurt_metric(
    model_name: str = "BLEURT-20",
) -> BleurtMetric:
    """Get the BLEURT metric instance.

    :param model_name: name of the COMET model to use
    :return: BLEURT metric instance
    """
    return BleurtMetric(model_name=model_name)
