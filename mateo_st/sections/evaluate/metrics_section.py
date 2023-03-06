import streamlit as st
from functions.metrics_constants import DEFAULT_METRICS, METRICS, SUPPORTED_LANGS_REV
from functions.metrics_functions import evaluate_input
from functions.utils import add_custom_pbar_style


def get_metrics_content():
    add_custom_pbar_style()

    content = st.container()
    content.markdown("## 📏 Metrics")
    has_others_transls = st.session_state["other_hyps"] and any(hyp for hyp in st.session_state["other_hyps"])
    if not (
        st.session_state["src_text"]
        and st.session_state["ref_text"]
        and (st.session_state["mt_text"] or has_others_transls)
    ):
        content.markdown(
            "Add the source text, reference translation and at least one other translation above to continue."
        )
        return None
    content.markdown(
        "Which metrics do you wish to use to evaluate the given MT output? For an overview of the"
        " available metrics and their supported languages, see the [metrics](metrics) page. Only metrics that"
        " are compatible with your soure/target language are shown."
    )

    select_col, eval_btn_col = content.columns((5, 1))

    # Make sure that we only allow/show metrics that are compatible. This means that:
    # - either the metric is not in the supported_lang dictionary (all the baseline metrics); or
    # - the target language key has to be in the list of the supported metrics, and in the case of comet also the src
    metric_options = [
        m
        for m in METRICS.keys()
        if m not in SUPPORTED_LANGS_REV
        or (
            st.session_state["tgt_lang_key"] in SUPPORTED_LANGS_REV[m]
            and (m != "comet" or st.session_state["src_lang_key"] in SUPPORTED_LANGS_REV[m])
        )
    ]
    selected_metrics = [m for m in DEFAULT_METRICS if m in metric_options]
    select_col.multiselect("Metrics to calculate", metric_options, selected_metrics, key="metrics")
    eval_btn_col.button("Evaluate!", on_click=evaluate_input, args=(content,))

    # Generate overview of given input
    overview = f"- **Source**: {st.session_state['src_text']}\n- **Ref.**: {st.session_state['ref_text']}"
    if st.session_state["mt_text"]:
        overview += f"\n- **MT**: {st.session_state['mt_text']}"

    for hyp_idx, other_hyp in enumerate(st.session_state["other_hyps"], 1):
        overview += f"\n- **Translation #{hyp_idx}**: {other_hyp}"

    content.markdown(overview)
