import streamlit as st
from functions.metrics_constants import METRICS_META, SUPPORTED_LANGS
from functions.utils import add_custom_base_style, add_custom_metrics_style


def main():
    st.set_page_config(page_title="Metrics overview | MATEO", page_icon="💯")
    add_custom_metrics_style()
    add_custom_base_style()

    st.title("💯 Metrics in MATEO")
    st.markdown(
        "We currently support the following evaluation metrics. A short description is given alongside"
        " references to their paper 📝, and the original underlying implementation 💻. We use"
        " [`evaluate`](https://github.com/huggingface/evaluate) as the main evaluation framework. In the future,"
        " more metrics will be added! If no supported languages are listed for a language, there is no restriction."
    )
    st.markdown(
        """
- 💡 baseline metric (fast, lower correlation with quality judgment by humans);
- 🚀 neural metric (slow, higher correlation).
"""
    )

    metrics_markdown = "<div className='metrics-wrapper'>"

    for metric_key, meta in METRICS_META.items():
        metrics_markdown += f"<div className='metric {meta['class']}'>"
        # header
        metrics_markdown += f"<header><h2>{meta['name']}</h2>"
        if "full_name" in meta and meta["full_name"]:
            metrics_markdown += f"<p>{meta['full_name']}</p>"
        metrics_markdown += "</header>"

        # Content
        metrics_markdown += f"<div className='metric-content'>{meta['description']}</div>"

        if "paper_url" in meta and meta["paper_url"]:
            metrics_markdown += (
                f"<div className='metric-paper'><p><a href='{meta['paper_url']}'"
                f" title='Paper of {meta['name']}' target='_blank'>Paper</a></p></div>"
            )

        # Implementation
        if "implementation" in meta and meta["implementation"]:
            metrics_markdown += f"<aside className='metric-implementation'>{meta['implementation']}</aside>"

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
