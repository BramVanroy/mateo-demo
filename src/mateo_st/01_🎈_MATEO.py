import streamlit as st
from mateo_st.utils import load_css


def main():
    st.set_page_config(page_title="MATEO: MAchine Translation Evaluation Online", page_icon="💯")
    load_css("base")

    st.title("🎈 MATEO: MAchine Translation Evaluation Online")

    st.markdown(
        "[MATEO](https://research.flw.ugent.be/projects/mateo-machine-translation-evaluation-online)"
        " (MAchine Translation Evaluation Online) is a project that is being developed at Ghent University, in the"
        " [Language and Translation Technology Team (LT3)](https://lt3.ugent.be/). It is funded by the European"
        " Association for Machine Translation and CLARIN.eu and will run until July 2023. It aims to bring automatic"
        " machine translation evaluation to the masses."
    )
    st.markdown("## Goals of the project")

    st.markdown(
        """
- Create an accessible website to evaluate machine translation for both experts and non-experts
- Create a Python package that focuses on machine translation that incorporates both baseline and state-of-the-art machine translation evaluation metrics
- Include research-focused functionality such as custom metric options, batch processing, exporting the results to formats such as LaTeX and Excel, and visualizing results in meaningful graphs
- Integrate the website in the CLARIN infrastructure, specifically the [CLARIN B centre of INT](https://portal.clarin.inl.nl/)
- Open-source *everything*

The current demo is a very first (alpha) indication of the direction that we are taking. Improvements that will follow:

- More metrics, both baseline metrics as newer ones. We will especially focus on metrics introduced in the last two years of the WMT Metric shared task
- Expanding beyond the sentence level. For a first demo, it makes sense to show quick sentence-level evaluations but to make the tool useful for research, it will also allow for corpus-level evaluation
- Python package and documentation will be made public, as well as the final project's source code
- More visualizations options on the website (e.g., visualizing edit distance)
- Integration in CLARIN infrastructure

Is there anything else you would like to see included? [Get in touch](#contact)!
"""
    )

    st.markdown("## 🏆 Funding")

    st.markdown(
        "MATEO is funded by a starting grant from the [European Association for Machine Translation](https://eamt.org/),"
        " and a full grant from [CLARIN.eu](https://www.clarin.eu/)'s _Bridging Gaps_ initiative."
    )

    st.markdown("## ✒️Contact")

    st.markdown(
        "Would you like  additional functionality in the final project? Do you want to collaborate? Or just want to get in touch?"
        " Give me a shout on [Twitter](https://twitter.com/BramVanroy) or"
        " [send an email](https://research.flw.ugent.be/nl/bram.vanroy)!"
    )


if __name__ == "__main__":
    main()
