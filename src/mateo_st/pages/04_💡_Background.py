import re

import streamlit as st
from mateo_st.metrics_constants import METRICS_META
from mateo_st.metrics_lang_support import SUPPORTED_LANGS
from mateo_st.utils import cli_args, load_css


def main():
    st.set_page_config(page_title="Background Information | MATEO", page_icon="💡")
    load_css("base")
    load_css("background")

    st.title("💡 Evaluation Information")

    st.header("Metrics in MATEO")
    st.markdown(
        "We currently support the following evaluation metrics. A short description is given alongside"
        " references to their paper 📝, and the original underlying implementation 💻. We use"
        " [`sacrebleu`](https://github.com/mjpost/sacrebleu) and"
        " [`evaluate`](https://github.com/huggingface/evaluate) as the main evaluation frameworks."
    )
    st.markdown(
        """
- 💡 baseline metric (faster, often a lower correlation with quality judgment by humans);
- 🚀 neural metric (slower, higher correlation).
"""
    )

    metrics_markdown = "<div className='metrics-wrapper'>"

    for metric_key, meta in METRICS_META.items():
        metrics_markdown += f"<div className='metric {meta.metric_class}'>"
        # header
        metrics_markdown += f"<header><h2>{meta.name}</h2>"
        if meta.full_name:
            metrics_markdown += f"<p>{meta.full_name}</p>"
        metrics_markdown += "</header>"

        # Content
        metrics_markdown += f"<div className='metric-content'>{meta.description_html}</div>"

        if meta.paper_url:
            metrics_markdown += (
                f"<div className='metric-paper'><p><a href='{meta.paper_url}'"
                f" title='Paper of {meta.name}' target='_blank'>Paper</a></p></div>"
            )

        # Implementation
        if meta.implementation_html:
            metrics_markdown += f"<aside className='metric-implementation'>{meta.implementation_html}</aside>"

        # Version
        if meta.version:
            version_str = meta.version
            # Add <code> tags if this is really just a version number, e.g. 0.1.2
            if re.match(r"^(?:\d+\.?){1,3}$", version_str):
                version_str = f"<code>{version_str}</code>"

            metrics_markdown += f"<aside className='metric-version'><p>Version {version_str}</p></aside>"

        # Supported languages
        if metric_key in SUPPORTED_LANGS and SUPPORTED_LANGS[metric_key]:
            metrics_markdown += "<aside className='metric-langs'><details><summary>Supported languages</summary><ul>"
            for lang, langcode in SUPPORTED_LANGS[metric_key].items():
                metrics_markdown += f"<li>{lang} (<code>{langcode}</code>)</li>"
            metrics_markdown += "</ul></details></aside>"

        metrics_markdown += "</div>"
    metrics_markdown += "</div>"

    st.markdown(metrics_markdown, unsafe_allow_html=True)

    st.header("Bootstrap resampling")

    st.info(
        'See ["Statistical Significance Tests for Machine Translation Evaluation"](https://aclanthology.org/W04-3250/) by P. Koehn for more information.',
        icon="💡",
    )
    st.markdown(
        """  
The bootstrap resampling as implemented in MATEO follows the implementation of ...


It is calculated more or less in this manner (baseline: bl; system: sys):
- calculate the "real" difference between bl and sys on the full corpus
- calculate scores for all n partitions (e.g. 1000) for the bl and sys. Partitions are drawn from the same set with
replacement. That means that if our dataset contains 300 samples, we create 1000 mini test sets of 300 samples that
are randomly chosen from our initial dataset of 300, but where a sample can occur multiple times. For motivation and
empirical evidence, see the aforementioned publication by Koehn
- calculate the absolute diff between the arrays of bl and system scores (result: array)
- subtract the mean from this array of absolute diffs. Now it indicates for each partition how "extreme" it is (how
different bl and sys are for this partition) compared to "the average partition"  
- find the number of cases where the absolute difference is larger ("more extreme") than the "real difference"
- divide this no. extreme cases by total no. cases (i.e. `n` partitions)

What we actually calculated is the probability that for a random subset (with replacement),
bl and sys differ more extremely than their real difference.

If this `p` value is high, then that means that extreme values (higher than full-corpus diff) are likely to occur.
In turn that also means that we can be _less certain_ that bl and sys _really_ differ significantly.

However, if the p value is low, then that means it is unlikely that for a random set, bl and sys differ
more extremely than for the full corpus (so partition scores are close to full-corpus scores).
That means that we can be more certain that bl and sys really differ significantly.

The 95% confidence interval that we can retrieve can be explained as "with a probability of 95%, the real mean
value of this metric for the full population that this dataset comes from, lies between `[mean-CI; mean+CI]`".
In other words, it tells you how close the calculated metric scores are for all different partitions.
"""
    )


if __name__ == "__main__":
    main()
    # Call this to immediately disable CUDA if needed
    cli_args()
