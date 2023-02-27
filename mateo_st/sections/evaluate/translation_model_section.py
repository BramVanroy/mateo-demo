from copy import copy

import streamlit as st
from functions.translator import TRANS_LANG2KEY, TRANS_SIZE2MODEL, Translator
from functions.utils import add_custom_input_style

from functions.utils import update_lang


def get_translation_model_content():
    content = st.container()
    add_custom_input_style()  # To make sure the swap button is aligned to the bottom

    content.markdown("## ✨ Translation model and language selection")
    content.markdown(
        "We will translate your source text automatically using Facebook's multilingual translation model"
        " [M2M100](https://ai.facebook.com/blog/introducing-many-to-many-multilingual-machine-translation/)"
        " behind the scenes. You can choose between two sizes, 418M (smaller) and 1.2B (larger, slower, better)."
    )
    content.markdown(
        "Even if you do not wish to make use of the automatic translation provided here, make sure to select"
        " the appropriate **source and target language**. Those will be used for evaluation too!"
    )

    # 2. Set source/target language
    def _swap_languages():
        old_src_lang = copy(st.session_state["src_lang"])
        st.session_state["src_lang"] = copy(st.session_state["tgt_lang"])
        st.session_state["tgt_lang"] = old_src_lang
        update_lang("src")
        update_lang("tgt")

    def _load_size_model():
        st.session_state["translator"] = Translator(
            src_lang=st.session_state["src_lang"],
            tgt_lang=st.session_state["tgt_lang"],
            no_cuda=st.session_state["no_cuda"],
        )

    src_lang_col, swap_btn_col, tgt_lang_col = content.columns((4, 1, 4))
    src_lang_col.selectbox(
        "Source language", tuple(TRANS_LANG2KEY.keys()), key="src_lang", on_change=update_lang, args=("src",)
    )
    swap_btn_col.button("⇄", on_click=_swap_languages)
    tgt_lang_col.selectbox(
        "Target language", tuple(TRANS_LANG2KEY.keys()), key="tgt_lang", on_change=update_lang, args=("tgt",)
    )

    load_info = content.info(
        f"(Down)loading a translation model for"
        f" {st.session_state['src_lang']} → {st.session_state['tgt_lang']}."
        f"\nThis may take a while the first time..."
    )

    if "translator" not in st.session_state or not st.session_state["translator"]:
        translator = None
        try:
            translator = Translator(
                src_lang=st.session_state["src_lang"],
                tgt_lang=st.session_state["tgt_lang"],
                no_cuda=st.session_state["no_cuda"],
            )
        except KeyError as exc:
            load_info.error(str(exc))

        if translator:
            st.session_state["translator"] = translator

    if "translator" in st.session_state:
        load_info.success(
            f"Translation model for {st.session_state['src_lang']} → {st.session_state['tgt_lang']} loaded!"
        )
