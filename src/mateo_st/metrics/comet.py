import comet
from mateo_st.metrics.base import MetricMeta, MetricOption


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
            name="config_name",
            description="COMET model to use. For supported models, look [here](https://github.com/Unbabel/COMET/tree/v1.1.3#comet-models).",
            default="eamt22-cometinho-da",
            choices=(
                "wmt20-comet-da",
                "wmt21-comet-da",
                "wmt21-cometinho-da",
                "eamt22-cometinho-da",
                "eamt22-prune-comet-da",
            ),
            demo_choices=(
                "wmt20-comet-da",
                "eamt22-cometinho-da",
            ),
        ),
    ),
    requires_source=True,
)