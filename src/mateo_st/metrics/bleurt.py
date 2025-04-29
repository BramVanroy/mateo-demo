import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict

from bleurt import score as bleurt_score
from huggingface_hub import snapshot_download
from mateo_st.metrics.base import MetricMeta, MetricOption, NeuralMetric


@dataclass
class BleurtMeta(MetricMeta):
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


bleurt_meta = BleurtMeta(
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
            name="config_name",
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
                "BLEURT-20-D3",
                "BLEURT-20-D6",
                "BLEURT-20-D12",
                "BLEURT-20",
            ),
        ),
    ),
)


@dataclass
class BleurtMetric(NeuralMetric):
    name = "bleurt"
    meta = bleurt_meta

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

    def compute(self, references: list[str], predictions: list[str], batch_size: int = 8) -> Any:
        """Predicts the score for a batch of references and hypotheses.

        :param references: list of reference sentences
        :param hypotheses: list of hypothesis sentences
        :param batch_size: batch size for processing
        :return: score result (dictionary)
        """
        print(f"Using {self.model_name} model for BLEURT scoring.")
        if len(references) != len(predictions):
            raise ValueError("The lengths of references and hypotheses must be the same.")

        return {
            "scores": self.model.score(
                references=references,
                candidates=predictions,
                batch_size=batch_size,
            )
        }
