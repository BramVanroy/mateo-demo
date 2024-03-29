from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict

from mateo_st.metrics.base import MetricMeta, MetricOption


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
