from mateo_st.metrics.base import MetricMeta, MetricOption
from sacrebleu import BLEU
from sacrebleu import __version__ as sb_version
from sacrebleu.metrics.bleu import _TOKENIZERS as SBTOKENIZERS


bleu_meta = MetricMeta(
    name="BLEU",
    metric_class="baseline",
    full_name="BiLingual Evaluation Understudy",
    description_html="<p>BLEU is perhaps the most well-known MT evaluation metric. It relies on how well a machine"
    " translation's n-grams correspond to those of a reference translation. Despite its popularity,"
    " it has also been criticized for its shortcomings such as a lack of sufficiently incorporating"
    " recall (e.g., <a href='https://aclanthology.org/E06-1032'"
    " title='Callison-Burch, Osborne, Koehn criticism on BLEU'>[1]</a>).</p>",
    paper_url="https://aclanthology.org/P02-1040/",
    implementation_html="<p><a href='https://github.com/mjpost/sacrebleu' title='SacreBLEU GitHub'>SacreBLEU</a></p>",
    is_default_selected=True,
    version=sb_version,
    sb_class=BLEU,
    options=(
        MetricOption(
            name="lowercase",
            description="If True, lowercased BLEU is computed.",
            default=False,
            types=(bool,),
        ),
        MetricOption(
            name="tokenize",
            description="The tokenizer to use",
            default=BLEU.TOKENIZER_DEFAULT,
            choices=tuple(SBTOKENIZERS.keys()),
        ),
        MetricOption(
            name="smooth_method",
            description="Smoothing method to use",
            default="exp",
            choices=("floor", "add-k", "exp", "none"),
        ),
        MetricOption(
            name="smooth_value",
            description="Smoothing value for `floor` and `add-k` methods. An empty value falls back to the default value",
            default="",
            types=(float, int),
            empty_str_is_none=True,
        ),
        MetricOption(
            name="max_ngram_order",
            description="The maximum n-gram order when computing precisions",
            default=4,
            types=(int,),
        ),
    ),
    segment_level=False,
    use_pseudo_batching=False,
)
