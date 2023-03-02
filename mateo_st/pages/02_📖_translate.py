from copy import copy
from io import StringIO
from math import ceil
from typing import Optional

import pandas as pd
import streamlit as st

from utils import set_general_session_keys, update_lang
from css import add_custom_translation_style, add_custom_base_style
from translator import TRANS_LANG2KEY, Translator, batch_translate


def _init():
    st.set_page_config(page_title="Translate | MATEO", page_icon="💯")
    add_custom_base_style()
    add_custom_translation_style()

    set_general_session_keys()
    if "translator" not in st.session_state:
        st.session_state["translator"] = None

    if "text" not in st.session_state:
        st.session_state["text"] = None

    if "stop_translating" not in st.session_state:
        st.session_state["stop_translating"] = False

    st.title("💯 Translate")
    st.markdown(
        "To provide quick access to multilingual translation, including for low-resource languages, we here provide"
        " access to Meta AI's open-source and open-access model [No Language Left Behind](https://ai.facebook.com/research/no-language-left-behind/)"
        " ([paper](https://arxiv.org/abs/2207.04672)). It enables translation to and from 200 languages."
    )


def _model_selection():
    st.markdown("## ✨ Language selection")

    def _swap_languages():
        st.session_state["text"] = None
        old_src_lang = copy(st.session_state["src_lang"])
        st.session_state["src_lang"] = copy(st.session_state["tgt_lang"])
        st.session_state["tgt_lang"] = old_src_lang
        update_lang("src")
        update_lang("tgt")

    src_lang_col, swap_btn_col, tgt_lang_col = st.columns((4, 1, 4))
    src_lang_col.selectbox(
        "Source language", tuple(TRANS_LANG2KEY.keys()), key="src_lang", on_change=update_lang, args=("src",)
    )
    swap_btn_col.button("⇄", on_click=_swap_languages)
    tgt_lang_col.selectbox(
        "Target language", tuple(TRANS_LANG2KEY.keys()), key="tgt_lang", on_change=update_lang, args=("tgt",)
    )

    load_info = st.info(
        f"(Down)loading a translation model for"
        f" {st.session_state['src_lang']} → {st.session_state['tgt_lang']}."
        f"\nThis may take a while the first time..."
    )

    if "translator" not in st.session_state or not st.session_state["translator"]:
        try:
            st.session_state["translator"] = Translator(
                src_lang=st.session_state["src_lang"],
                tgt_lang=st.session_state["tgt_lang"],
                no_cuda=st.session_state["no_cuda"],
            )
        except KeyError as exc:
            load_info.error(str(exc))

    if "translator" in st.session_state:
        load_info.success(
            f"Translation model for {st.session_state['src_lang']} → {st.session_state['tgt_lang']} loaded!"
        )


def _data_input():
    inp_data_heading, input_col = st.columns((3, 1))
    inp_data_heading.markdown("## Input data 📄")

    fupload_check = input_col.checkbox("File upload?")

    st.markdown("Make sure that the file or text box contains **one sentence per line**. Empty lines will be removed.")
    if fupload_check:
        uploaded_file = st.file_uploader("Text file", label_visibility="hidden")
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            st.session_state["text"] = stringio.read()
        else:
            st.session_state["text"] = None
    else:
        st.session_state["text"] = st.text_area(label="Sentences to translate", label_visibility="hidden")


def _get_increment_size(num_sents) -> int:
    if st.session_state["transl_batch_size"] >= num_sents:
        return 100
    else:
        return ceil(100 / (num_sents / st.session_state["transl_batch_size"]))


def _translate():
    if "text" not in st.session_state or not st.session_state["text"]:
        return None
    elif "translator" in st.session_state and st.session_state["translator"]:
        st.markdown("## Translations")
        transl_info = st.info("Translating...")

        pbar = st.progress(0)
        sentences = [s.strip() for s in st.session_state["text"].splitlines() if s.strip()]
        num_sentences = len(sentences)
        increment = _get_increment_size(num_sentences)
        percent_done = 0
        all_translations = []

        transl_ct = st.empty()
        download_btn_ct = st.empty()
        for translations in batch_translate(st.session_state["translator"],
                                            sentences,
                                            batch_size=st.session_state["transl_batch_size"]):
            all_translations.extend(translations)

            df = pd.DataFrame(list(zip(sentences, all_translations)), columns=["src", "mt"])
            transl_ct.table(df)
            percent_done += increment
            pbar.progress(min(percent_done, 100))

        download_btn_ct.download_button(
            "Download translations",
            "\n".join(all_translations) + "\n",
            "translations.txt",
            "text",
            key="download-txt",
            help="Download your translated text"
        )
        pbar.empty()
        transl_info.info("Done translating! You can download the translations at the end of this page.")


def main():
    _init()
    _model_selection()
    _data_input()
    _translate()


if __name__ == "__main__":
    main()
