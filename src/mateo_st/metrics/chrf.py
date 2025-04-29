from sacrebleu import CHRF
from sacrebleu import __version__ as sb_version

from mateo_st.metrics.base import MetricMeta, MetricOption


chrf_meta = MetricMeta(
    name="ChrF",
    metric_class="baseline",
    full_name="Character F-score",
    description_html="<p>ChrF uses the F-score statistic for character n-gram matches. It therefore focuses on"
    " characters rather than words. It has shown to be a strong baseline, often outperforming other"
    " baselines like BLEU or TER. By default, up to character 6-grams are considered.</p>",
    paper_url="https://aclanthology.org/W15-3049",
    implementation_html="<p><a href='https://github.com/mjpost/sacrebleu' title='SacreBLEU GitHub'>SacreBLEU</a></p>",
    is_default_selected=True,
    version=sb_version,
    sb_class=CHRF,
    options=(
        MetricOption(name="char_order", description="Character n-gram order", default=CHRF.CHAR_ORDER, types=(int,)),
        MetricOption(
            name="word_order",
            description="Word n-gram order. If equals to 2, the metric is referred to as chrF++",
            default=CHRF.WORD_ORDER,
            types=(int,),
        ),
        MetricOption(
            name="beta",
            description="Determines the importance of recall w.r.t precision",
            default=CHRF.BETA,
            types=(int,),
        ),
        MetricOption(name="lowercase", description="Whether to lowercase the data", default=False, types=(bool,)),
        MetricOption(
            name="whitespace",
            description="Whether to include whitespaces when extracting character n-grams",
            default=False,
            types=(bool,),
        ),
        MetricOption(
            name="eps_smoothing",
            description="Whether to apply epsilon smoothing similar to reference chrF++.py, NLTK and Moses implementations",
            default=False,
            types=(bool,),
        ),
    ),
    segment_level=False,
)
