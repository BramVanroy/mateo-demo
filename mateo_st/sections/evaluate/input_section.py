from urllib.parse import quote

import streamlit as st
from functions.utils import add_custom_input_style


def get_input_content():
    add_custom_input_style()

    content = st.container()

    # SOURCE/REF/HYPS
    content.text_area(label="Source text", key="src_text")
    content.text_area(label="Reference translation", key="ref_text")
    content.text_area(label="Machine translation", key="mt_text")

    # Other hypotheses
    other_hyps_ct = content.container()
    sl = st.session_state["src_lang_key"]
    tl = st.session_state["tgt_lang_key"]
    encoded_src_text = quote(st.session_state["src_text"])
    gtrans_url = f"https://translate.google.com/?sl={sl}&tl={tl}&text={encoded_src_text}&op=translate"
    deepl_url = f"https://www.deepl.com/translator#{sl}/{tl}/{encoded_src_text}"
    other_hyps_ct.markdown(
        "In addition to the automatically generated machine translation above, you can also add other"
        f" translations. For instance from [Google Translate]({gtrans_url}), [DeepL]({deepl_url}), or your own."
    )

    # Adding/remove additional hypothesis fields
    def _add_transl_field():
        st.session_state["other_hyps"].append("")

    def _remove_transl_field(idx):
        del st.session_state["other_hyps"][idx]

    def _update_transl_field(idx):
        st.session_state["other_hyps"][idx] = st.session_state[f"other_hyps_{idx}"]

    other_hyps_ct.button(
        "Add translation", disabled=len(st.session_state["other_hyps"]) == 2, on_click=_add_transl_field
    )

    for transl_id, transl in enumerate(st.session_state["other_hyps"]):
        transl_text_col, transl_rm_btn_col = other_hyps_ct.columns((5, 1))
        transl_text_col.text_area(
            f"Other translation #{transl_id + 1}",
            on_change=_update_transl_field,
            args=(transl_id,),
            value=transl,
            key=f"other_hyps_{transl_id}",
        )
        transl_rm_btn_col.button(
            "Remove", on_click=_remove_transl_field, args=(transl_id,), key=f"remove_btn_{transl_id}"
        )
