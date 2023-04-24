from copy import copy
from io import StringIO
from math import ceil

import numpy as np
import pandas as pd
import streamlit as st
from mateo_st.css import add_custom_base_style, add_custom_translation_style
from mateo_st.translator import TRANS_LANG2KEY, TRANS_SIZE2MODEL, Translator
from mateo_st.utils import cli_args, create_download_link, set_general_session_keys, update_translator_lang


def _init():
    st.set_page_config(page_title="Automatically Translate | MATEO", page_icon="📖")
    add_custom_base_style()
    add_custom_translation_style()

    set_general_session_keys()

    if "translator" not in st.session_state:
        st.session_state["translator"] = None

    if "text_to_translate" not in st.session_state:
        st.session_state["text_to_translate"] = None

    st.title("📖 Translate")
    st.markdown(
        "To provide quick access to multilingual translation, including for low-resource languages, we here provide"
        " access to Meta AI's open-source and open-access model "
        "[No Language Left Behind](https://ai.facebook.com/research/no-language-left-behind/)"
        " ([paper](https://arxiv.org/abs/2207.04672)). It enables machine translation to and from 200 languages."
        " In this interface, we specifically use"
        f" [{st.session_state['transl_model_size']}](https://huggingface.co/{TRANS_SIZE2MODEL[st.session_state['transl_model_size']]})"
        f" (max. length: {cli_args().transl_max_length:,}; num. beams: {cli_args().transl_num_beams};"
        f" batch size: {cli_args().transl_batch_size})."
    )


def _model_selection():
    st.markdown("## ✨ Language selection")

    def _swap_languages():
        st.session_state["text_to_translate"] = None
        old_src_lang = copy(st.session_state["src_lang"])
        st.session_state["src_lang"] = copy(st.session_state["tgt_lang"])
        st.session_state["tgt_lang"] = old_src_lang
        update_translator_lang("src")
        update_translator_lang("tgt")

    src_lang_col, swap_btn_col, tgt_lang_col = st.columns((4, 1, 4))
    src_lang_col.selectbox(
        "Source language",
        tuple(TRANS_LANG2KEY.keys()),
        key="src_lang",
        on_change=update_translator_lang,
        args=("src",),
    )
    swap_btn_col.button("⇄", on_click=_swap_languages)
    tgt_lang_col.selectbox(
        "Target language",
        tuple(TRANS_LANG2KEY.keys()),
        key="tgt_lang",
        on_change=update_translator_lang,
        args=("tgt",),
    )

    load_info = st.info(
        f"(Down)loading model {st.session_state['transl_model_size']} for"
        f" {st.session_state['src_lang']} → {st.session_state['tgt_lang']}."
        f"\nThis may take a while the first time..."
    )

    if "translator" not in st.session_state or not st.session_state["translator"]:
        try:
            st.session_state["translator"] = Translator(
                src_lang=st.session_state["src_lang"],
                tgt_lang=st.session_state["tgt_lang"],
                model_size=st.session_state["transl_model_size"],
                max_length=st.session_state["transl_max_length"],
                num_beams=st.session_state["transl_num_beams"],
                no_cuda=st.session_state["transl_no_cuda"] or st.session_state["no_cuda"],
            )
        except KeyError as exc:
            load_info.error(str(exc))

    if "translator" in st.session_state and st.session_state["translator"] is not None:
        load_info.success(
            f"Translation model {st.session_state['translator'].model_name.split('/')[-1]} loaded for"
            f" {st.session_state['src_lang']} → {st.session_state['tgt_lang']}!"
        )


def _data_input():
    inp_data_heading, input_col = st.columns((3, 1))
    inp_data_heading.markdown("## Input data 📄")

    fupload_check = input_col.checkbox("File upload?")

    st.markdown(
        "Make sure that the file or text in the text box contains **one sentence per line**. Empty lines will"
        " be removed."
    )
    if fupload_check:
        uploaded_file = st.file_uploader("Text file", label_visibility="hidden")
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            st.session_state["text_to_translate"] = stringio.read()
        else:
            st.session_state["text_to_translate"] = None
    else:
        st.session_state["text_to_translate"] = st.text_area(label="Sentences to translate", label_visibility="hidden")


def _get_increment_size(num_sents) -> int:
    if st.session_state["transl_batch_size"] >= num_sents:
        return 100
    else:
        return ceil(100 / (num_sents / st.session_state["transl_batch_size"]))


def _translate():
    if "text_to_translate" not in st.session_state or not st.session_state["text_to_translate"]:
        return None
    elif "translator" in st.session_state and st.session_state["translator"]:
        st.markdown("## Translations")
        transl_info = st.info("Translating...")
        download_info = st.empty()

        pbar = st.progress(0)
        sentences = [s.strip() for s in st.session_state["text_to_translate"].splitlines() if s.strip()]
        num_sentences = len(sentences)
        increment = _get_increment_size(num_sentences)
        percent_done = 0
        all_translations = []

        transl_ct = st.empty()
        df = pd.DataFrame()
        for translations in st.session_state["translator"].batch_translate(
            sentences, batch_size=st.session_state["transl_batch_size"]
        ):
            all_translations.extend(translations)

            df = pd.DataFrame(
                list(zip(sentences, all_translations)),
                columns=[
                    f"src ({st.session_state['src_lang_key']})",
                    f"mt ({st.session_state['tgt_lang_key']}):"
                    f" {st.session_state['translator'].model_name.split('/')[-1]}",
                ],
            )
            df.index = np.arange(1, len(df) + 1)  # Index starting at number 1
            transl_ct.dataframe(df)
            percent_done += increment
            pbar.progress(min(percent_done, 100))

        pbar.empty()
        transl_info.success("Done translating!")

        xlsx_download_html = create_download_link(df, "translations.xlsx", "Download Excel")
        txt_download_html = create_download_link(
            "\n".join(all_translations) + "\n", "translations.txt", "Download text"
        )
        download_info.markdown(
            f"- <strong>{xlsx_download_html}</strong>: parallel source/MT sentences as Excel file;\n"
            f"- <strong>{txt_download_html}</strong>: only translations, as a plain text file.",
            unsafe_allow_html=True,
        )


def main():
    _init()
    _model_selection()
    _data_input()
    _translate()


if __name__ == "__main__":
    main()
