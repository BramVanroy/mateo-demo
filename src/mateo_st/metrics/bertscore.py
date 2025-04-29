from dataclasses import dataclass, field
from statistics import mean
from typing import Any, ClassVar

import bert_score
import streamlit as st

from mateo_st.metrics.base import MetricMeta, MetricOption, NeuralMetric


bertscore_meta = MetricMeta(
    name="BERTScore",
    metric_class="neural",
    full_name="",
    description_html="BERTScore is an automatic evaluation metric for text generation that computes similarity scores"
    " for each token in the machine translation with each token in the reference translation. These"
    " similarity scores are based on contextual embeddings retrieved from a"
    " <a href='https://aclanthology.org/N19-1423/' title='BERT paper'>BERT</a>"
    " language model.</p>",
    paper_url="https://openreview.net/forum?id=SkeHuCVFDr",
    implementation_html="<p><a href='https://github.com/Tiiiger/bert_score' title='BertScore GitHub'"
    ">BERTScore</a></p>",
    version=bert_score.__version__,
    corpus_score_key="mean_f1",  # Manually added in postprocessing
    sentences_score_key="f1",
    options=(
        MetricOption(
            name="lang",
            description="Language of the translations. This is an optional shortcut, used to select a good default"
            " model for your language"
            " (en: roberta-large, zh: bert-base-chinese, tr: dbmdz/bert-base-turkish-cased, en-sci:"
            " allenai/scibert_scivocab_uncased, and bert-base-multilingual-cased for all 'other').\n"
            " Alternatively, choose a model from the 'model_type' option. By default 'other' is selected, with works"
            " with many different languages.\n⚠️ 'model_type' has precedence over 'lang' so make sure to set"
            " 'model_type' to '' when selecting a 'lang'!",
            default="other",
            choices=("other", "en", "zh", "tr", "en-sci"),
        ),
        MetricOption(
            name="model_type",
            description="BERTScore model to use. Benchmarked scores on to-English WMT data can be found [here](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0).",
            default="",
            choices=(
                "",
                "bert-base-uncased",
                "bert-large-uncased",
                "bert-base-cased-finetuned-mrpc",
                "bert-base-multilingual-cased",
                "bert-base-chinese",
                "roberta-base",
                "roberta-large",
                "roberta-large-mnli",
                "roberta-base-openai-detector",
                "roberta-large-openai-detector",
                "xlnet-base-cased",
                "xlnet-large-cased",
                "xlm-mlm-en-2048",
                "xlm-mlm-100-1280",
                "allenai/scibert_scivocab_uncased",
                "allenai/scibert_scivocab_cased",
                "nfliu/scibert_basevocab_uncased",
                "distilroberta-base",
                "distilbert-base-uncased",
                "distilbert-base-uncased-distilled-squad",
                "distilbert-base-multilingual-cased",
                "albert-base-v1",
                "albert-large-v1",
                "albert-xlarge-v1",
                "albert-xxlarge-v1",
                "albert-base-v2",
                "albert-large-v2",
                "albert-xlarge-v2",
                "albert-xxlarge-v2",
                "xlm-roberta-base",
                "xlm-roberta-large",
                "google/electra-small-generator",
                "google/electra-small-discriminator",
                "google/electra-base-generator",
                "google/electra-base-discriminator",
                "google/electra-large-generator",
                "google/electra-large-discriminator",
                "google/bert_uncased_L-2_H-128_A-2",
                "google/bert_uncased_L-2_H-256_A-4",
                "google/bert_uncased_L-2_H-512_A-8",
                "google/bert_uncased_L-2_H-768_A-12",
                "google/bert_uncased_L-4_H-128_A-2",
                "google/bert_uncased_L-4_H-256_A-4",
                "google/bert_uncased_L-4_H-512_A-8",
                "google/bert_uncased_L-4_H-768_A-12",
                "google/bert_uncased_L-6_H-128_A-2",
                "google/bert_uncased_L-6_H-256_A-4",
                "google/bert_uncased_L-6_H-512_A-8",
                "google/bert_uncased_L-6_H-768_A-12",
                "google/bert_uncased_L-8_H-128_A-2",
                "google/bert_uncased_L-8_H-256_A-4",
                "google/bert_uncased_L-8_H-512_A-8",
                "google/bert_uncased_L-8_H-768_A-12",
                "google/bert_uncased_L-10_H-128_A-2",
                "google/bert_uncased_L-10_H-256_A-4",
                "google/bert_uncased_L-10_H-512_A-8",
                "google/bert_uncased_L-10_H-768_A-12",
                "google/bert_uncased_L-12_H-128_A-2",
                "google/bert_uncased_L-12_H-256_A-4",
                "google/bert_uncased_L-12_H-512_A-8",
                "google/bert_uncased_L-12_H-768_A-12",
                "amazon/bort",
                "facebook/bart-base",
                "facebook/bart-large",
                "facebook/bart-large-cnn",
                "facebook/bart-large-mnli",
                "facebook/bart-large-xsum",
                "t5-small",
                "t5-base",
                "t5-large",
                "vinai/bertweet-base",
                "microsoft/deberta-base",
                "microsoft/deberta-base-mnli",
                "microsoft/deberta-large",
                "microsoft/deberta-large-mnli",
                "microsoft/deberta-xlarge",
                "microsoft/deberta-xlarge-mnli",
                "YituTech/conv-bert-base",
                "YituTech/conv-bert-small",
                "YituTech/conv-bert-medium-small",
                "microsoft/mpnet-base",
                "squeezebert/squeezebert-uncased",
                "squeezebert/squeezebert-mnli",
                "squeezebert/squeezebert-mnli-headless",
                "tuner007/pegasus_paraphrase",
                "google/pegasus-large",
                "google/pegasus-xsum",
                "sshleifer/tiny-mbart",
                "facebook/mbart-large-cc25",
                "facebook/mbart-large-50",
                "facebook/mbart-large-en-ro",
                "facebook/mbart-large-50-many-to-many-mmt",
                "facebook/mbart-large-50-one-to-many-mmt",
                "allenai/led-base-16384",
                "facebook/blenderbot_small-90M",
                "facebook/blenderbot-400M-distill",
                "microsoft/prophetnet-large-uncased",
                "microsoft/prophetnet-large-uncased-cnndm",
                "SpanBERT/spanbert-base-cased",
                "SpanBERT/spanbert-large-cased",
                "microsoft/xprophetnet-large-wiki100-cased",
                "ProsusAI/finbert",
                "Vamsi/T5_Paraphrase_Paws",
                "ramsrigouthamg/t5_paraphraser",
                "microsoft/deberta-v2-xlarge",
                "microsoft/deberta-v2-xlarge-mnli",
                "microsoft/deberta-v2-xxlarge",
                "microsoft/deberta-v2-xxlarge-mnli",
                "allenai/longformer-base-4096",
                "allenai/longformer-large-4096",
                "allenai/longformer-large-4096-finetuned-triviaqa",
                "zhiheng-huang/bert-base-uncased-embedding-relative-key",
                "zhiheng-huang/bert-base-uncased-embedding-relative-key-query",
                "zhiheng-huang/bert-large-uncased-whole-word-masking-embedding-relative-key-query",
                "google/mt5-small",
                "google/mt5-base",
                "google/mt5-large",
                "google/mt5-xl",
                "google/bigbird-roberta-base",
                "google/bigbird-roberta-large",
                "google/bigbird-base-trivia-itc",
                "princeton-nlp/unsup-simcse-bert-base-uncased",
                "princeton-nlp/unsup-simcse-bert-large-uncased",
                "princeton-nlp/unsup-simcse-roberta-base",
                "princeton-nlp/unsup-simcse-roberta-large",
                "princeton-nlp/sup-simcse-bert-base-uncased",
                "princeton-nlp/sup-simcse-bert-large-uncased",
                "princeton-nlp/sup-simcse-roberta-base",
                "princeton-nlp/sup-simcse-roberta-large",
                "dbmdz/bert-base-turkish-cased",
                "dbmdz/distilbert-base-turkish-cased",
                "google/byt5-small",
                "google/byt5-base",
                "google/byt5-large",
                "microsoft/deberta-v3-xsmall",
                "microsoft/deberta-v3-small",
                "microsoft/deberta-v3-base",
                "microsoft/mdeberta-v3-base",
                "microsoft/deberta-v3-large",
                "khalidalt/DeBERTa-v3-large-mnli",
            ),
            demo_choices=(
                "",
                "roberta-large",
                "bert-base-chinese",
                "dbmdz/bert-base-turkish-cased",
                "allenai/scibert_scivocab_uncased",
                "bert-base-multilingual-cased",
            ),
            empty_str_is_none=True,
        ),
        MetricOption(
            name="num_layers",
            description="This layer's representation will be used. If empty, defaults to the best layer as tuned"
            " on WMT16",
            default="",
            types=(int,),
            empty_str_is_none=True,
        ),
        MetricOption(
            name="batch_size",
            description="How many sentences to process at once. The larger the batch size, the faster the scoring but the higher the memory usage. Do not change this unless you know what you are doing. If the value is set too high it will freeze your computer and potentially crash the app!",
            default=4,
            types=(int,),
            disabled="auto",
            is_init_arg=False,
        ),
        # Not adding other options such as rescale_with_baseline or idf, because those require extra corpus input
        # to calculate baseline/idf scores on
    ),
)


@dataclass
class BertScoreMetric(NeuralMetric):
    name: ClassVar[str] = "bertscore"
    meta: ClassVar[MetricMeta] = bertscore_meta

    lang: str | None = field(
        default=None,
        metadata={
            "help": "Language of the translations. This is an optional shortcut, used to select a good default model for your language"
        },
    )
    model_type: str | None = field(
        default=None,
        metadata={
            "help": "BERTScore model to use. Benchmarked scores on to-English WMT data can be found [here](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0)."
        },
    )
    num_layers: int | None = field(
        default=None,
        metadata={
            "help": "This layer's representation will be used. If empty, defaults to the best layer as tuned on WMT16"
        },
    )

    model: bert_score.BERTScorer = field(default=None, init=False)

    def __post_init__(self):
        if self.model_type is None:
            if self.lang is None:
                raise ValueError(
                    "Either 'lang' (e.g. 'en') or 'model_type' (e.g. 'microsoft/deberta-xlarge-mnli')"
                    " must be specified"
                )
            self.model_type = bert_score.utils.lang2model[self.lang.lower()]

        if self.num_layers is None:
            self.num_layers = bert_score.utils.model2layers[self.model_type]

        self.model = bert_score.BERTScorer(
            model_type=self.model_type,
            num_layers=self.num_layers,
            lang=self.lang,
        )

    def compute(
        self,
        references: list[str],
        predictions: list[str],
        batch_size: int = 16,
        **kwargs,
    ) -> dict:
        """Predicts the score for a batch of references and hypotheses. BertScore is a bit different from the others in that initialization happens inside
        this compute function but the scorer is cached for the next call.

        :param references: list of reference sentences
        :param hypotheses: list of hypothesis sentences
        :param lang: language of the translations. This is an optional shortcut, used to select a good default model for your language
        :param model_type: BERTScore model to use. Benchmarked scores on to-English WMT data can be found [here](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0).
        :param num_layers: This layer's representation will be used. If empty, defaults to the best layer as tuned on WMT16
        :param batch_size: batch size for scoring
        :param kwargs: additional arguments for the model prediction
        :return: score result (dictionary)
        """
        if len(references) != len(predictions):
            raise ValueError("The lengths of references and hypotheses must be the same.")

        P, R, F = self.model.score(
            cands=predictions,
            refs=references,
            verbose=False,
            batch_size=batch_size,
            **kwargs,
        )

        output_dict = {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F.tolist(),
        }
        return output_dict

    @classmethod
    def postprocess_result(cls, result: dict[str, Any]):
        """Post-processes the result that is retrieved from a computed metric.

        :param result: score result (dictionary)
        :return: modified score result
        """
        corpus_key = cls.meta.corpus_score_key
        sentences_key = cls.meta.sentences_score_key
        result[corpus_key] = 100 * mean(result["f1"])
        result[sentences_key] = [score * 100 for score in result["f1"]]
        return result


@st.cache_resource(show_spinner=False, max_entries=2, ttl=60 * 60 * 24 * 30)
def get_bertscore_metric(
    lang: str | None = None,
    model_type: str | None = None,
    num_layers: int | None = None,
) -> BertScoreMetric:
    """Get a BERTScore metric instance.

    :param lang: Language of the translations. This is an optional shortcut, used to select a good default model for your language
    :param model_type: BERTScore model to use. Benchmarked scores on to-English WMT data can be found [here](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0).
    :param num_layers: This layer's representation will be used. If empty, defaults to the best layer as tuned on WMT16
    :param device: Device to use for scoring (e.g. "cuda", "cpu"). If None, will use the default device.
    :return: BERTScore metric instance
    """
    return BertScoreMetric(
        lang=lang,
        model_type=model_type,
        num_layers=num_layers,
    )
