from mateo_st.metrics.base import MetricMeta, MetricOption
from sacrebleu import TER
from sacrebleu import __version__ as sb_version


ter_meta = MetricMeta(
    name="TER",
    metric_class="baseline",
    full_name="Translation Edit Rate",
    description_html="TER is an intuitive baseline metric that formulates MT quality as the number of edits required"
    " to change an MT translation so that it exactly matches a reference translation. The possible"
    " edit operations are insertions, deletions, substitutions of words as well as word sequence"
    " ('phrase') shifts.</p>",
    paper_url="https://aclanthology.org/2006.amta-papers.25/",
    implementation_html="<p><a href='https://github.com/mjpost/sacrebleu' title='SacreBLEU GitHub'>SacreBLEU</a></p>",
    higher_better=False,
    version=sb_version,
    sb_class=TER,
    options=(
        MetricOption(
            name="normalized",
            description="Whether to enable character normalization. If enabled, by default, normalizes a couple of things such as newlines being stripped, retrieving XML encoded characters, and fixing tokenization for punctuation. When 'asian_support' is enabled, also normalizes specific Asian (CJK) character sequences, i.e. split them down to the character level.",
            default=False,
            types=(bool,),
        ),
        MetricOption(
            name="no_punct",
            description="Whether to removes punctuations from sentences. Can be used in conjunction with 'asian_support' to also remove typical punctuation markers in Asian languages (CJK)",
            default=False,
            types=(bool,),
        ),
        MetricOption(
            name="asian_support",
            description="Enable special treatment of Asian characters. This option only has an effect when 'normalized' and/or 'no_punct' is enabled. If 'normalized' is also enabled, then Asian (CJK) characters are split down to the character level. If 'no_punct' is enabled alongside 'asian_support', specific unicode ranges for CJK and full-width punctuations are also removed.",
            default=False,
            types=(bool,),
        ),
        MetricOption(
            name="case_sensitive", description="Whether to NOT lowercase the data", default=False, types=(bool,)
        ),
    ),
    segment_level=False,
    use_pseudo_batching=False,
)
