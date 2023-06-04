import bert_score
from mateo_st.metrics.base import MetricMeta, MetricOption


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
            " Alternatively, choose a model from the 'model_type' option.\n⚠️ 'model_type' has"
            " precedence over 'lang' so make sure to set 'model_type' to '' when selecting a 'lang'!",
            default="other",
            choices=("other", "en", "zh", "tr", "en-sci"),
        ),
        MetricOption(
            name="model_type",
            description="BERTScore model to use. Benchmarked scores on to-English WMT data can be found [here](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0).",
            default="",
            # Not all models because we want to save compute
            # All options here: https://github.com/Tiiiger/bert_score/blob/dbcf6db37e8bd6ff68446f06b0ba5d0763b62d20/bert_score/utils.py#L40
            choices=(
                "",
                "bert-base-multilingual-cased",
                "roberta-base",
                "roberta-large",
                "bert-base-chinese",
                "dbmdz/bert-base-turkish-cased",
                "allenai/scibert_scivocab_uncased",
                "facebook/bart-base",
                "princeton-nlp/sup-simcse-bert-base-uncased",
                "microsoft/deberta-v3-base",
                "microsoft/deberta-v3-large",
                "microsoft/mdeberta-v3-base",
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
        )
        # Not adding other options such as rescale_with_baseline or idf, because those require extra corpus input
        # to calculate baseline/idf scores on
    ),
)
