import re

import streamlit as st
from mateo_st.metrics_constants import METRICS_META
from mateo_st.metrics_lang_support import SUPPORTED_LANGS
from mateo_st.utils import cli_args, load_css


def main():
    st.set_page_config(page_title="Metrics Overview | MATEO", page_icon="💡")
    load_css("base")
    load_css("metrics")

    st.title("💡 Metrics in MATEO")
    st.markdown(
        "We currently support the following evaluation metrics. A short description is given alongside"
        " references to their paper 📝, and the original underlying implementation 💻. We use"
        " [`evaluate`](https://github.com/huggingface/evaluate) as the main evaluation framework."
    )
    st.markdown(
        """
- 💡 baseline metric (faster, lower correlation with quality judgment by humans);
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


if __name__ == "__main__":
    main()
    # Call this to immediately disable CUDA if needed
    cli_args()
