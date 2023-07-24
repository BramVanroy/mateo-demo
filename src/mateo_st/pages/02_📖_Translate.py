from copy import copy
from io import StringIO
from math import ceil

import numpy as np
import pandas as pd
import streamlit as st
from mateo_st.translator import TRANS_LANG2KEY, TRANS_SIZE2MODEL, Translator, update_translator_lang
from mateo_st.utils import cli_args, create_download_link, load_css


def _init():
    st.set_page_config(page_title="Automatically Translate | MATEO", page_icon="📖")
    load_css("base")
    load_css("translate")

    if "translator" not in st.session_state:
        st.session_state["translator"] = None

    if "text_to_translate" not in st.session_state:
        st.session_state["text_to_translate"] = None

    # LANGUAGES
    if "src_lang" not in st.session_state:
        st.session_state["src_lang"] = "English"
    elif st.session_state["src_lang"] not in TRANS_LANG2KEY:
        st.session_state["src_lang"] = "English"

    if "src_lang_key" not in st.session_state:
        st.session_state["src_lang_key"] = TRANS_LANG2KEY[st.session_state["src_lang"]]

    if "tgt_lang" not in st.session_state:
        st.session_state["tgt_lang"] = "Dutch"
    elif st.session_state["tgt_lang"] not in TRANS_LANG2KEY:
        st.session_state["tgt_lang"] = "Dutch"

    if "tgt_lang_key" not in st.session_state:
        st.session_state["tgt_lang_key"] = TRANS_LANG2KEY[st.session_state["tgt_lang"]]

    st.title("📖 Translate")
    st.markdown(
        "To provide quick access to multilingual translation, including for low-resource languages, we here provide"
        " access to Meta AI's open-source and open-access model "
        "[No Language Left Behind](https://ai.facebook.com/research/no-language-left-behind/)"
        " ([paper](https://arxiv.org/abs/2207.04672)). It enables machine translation to and from 200 languages. In"
        f" this interface, we specifically use"
        f" [{cli_args().transl_model_size}](https://huggingface.co/{TRANS_SIZE2MODEL[cli_args().transl_model_size]})"
        f" with a maximal length of {cli_args().transl_max_length} and {cli_args().transl_num_beams} beam(s)."
    )

    with st.expander("✒️ If you use MATEO for your work, please **cite it** accordingly."):
        st.markdown(
            """> Vanroy, B., Tezcan, A., & Macken, L. (2023). MATEO: MAchine Translation Evaluation Online. In M. Nurminen, J. Brenner, M. Koponen, S. Latomaa, M. Mikhailov, F. Schierl, … H. Moniz (Eds.), _Proceedings of the 24th Annual Conference of the European Association for Machine Translation_ (pp. 499–500). Tampere, Finland: European Association for Machine Translation (EAMT).

```bibtex
@inproceedings{vanroy2023mateo,
    author       = {{Vanroy, Bram and Tezcan, Arda and Macken, Lieve}},
    booktitle    = {{Proceedings of the 24th Annual Conference of the European Association for Machine Translation}},
    editor       = {{Nurminen, Mary and Brenner, Judith and Koponen, Maarit and Latomaa, Sirkku and Mikhailov, Mikhail and Schierl, Frederike and Ranasinghe, Tharindu and Vanmassenhove, Eva and Alvarez Vidal, Sergi and Aranberri, Nora and Nunziatini, Mara and Parra Escartín, Carla and Forcada, Mikel and Popovic, Maja and Scarton, Carolina and Moniz, Helena}},
    isbn         = {{978-952-03-2947-1}},
    language     = {{eng}},
    location     = {{Tampere, Finland}},
    pages        = {{499--500}},
    publisher    = {{European Association for Machine Translation (EAMT)}},
    title        = {{MATEO: MAchine Translation Evaluation Online}},
    url          = {{https://lt3.ugent.be/mateo/}},
    year         = {{2023}},
}
```"""
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
        f"(Down)loading model {cli_args().transl_model_size} for"
        f" {st.session_state['src_lang']} → {st.session_state['tgt_lang']}."
        f"\nThis may take a while the first time..."
    )

    if "translator" not in st.session_state or not st.session_state["translator"]:
        try:
            st.session_state["translator"] = Translator(
                src_lang=st.session_state["src_lang"],
                tgt_lang=st.session_state["tgt_lang"],
                model_size=cli_args().transl_model_size,
                quantize=not cli_args().transl_no_quantize,
                no_cuda=cli_args().transl_no_cuda or cli_args().no_cuda,
            )
        except KeyError as exc:
            load_info.exception(exc)

    if "translator" in st.session_state and st.session_state["translator"] is not None:
        load_info.success(
            f"Translation model {st.session_state['translator'].model_name.split('/')[-1]} loaded for"
            f" {st.session_state['src_lang']} → {st.session_state['tgt_lang']}!"
        )


def _data_input():
    inp_data_heading, input_col = st.columns((3, 1))
    inp_data_heading.markdown("## 📄 Input data")

    fupload_check = input_col.checkbox("File upload?")

    st.markdown(
        "Make sure that the file or text in the text box contains **one sentence per line**. Empty lines will"
        " be removed."
    )
    if fupload_check:
        uploaded_file = st.file_uploader("Text file", label_visibility="collapsed")
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            st.session_state["text_to_translate"] = stringio.read()
        else:
            st.session_state["text_to_translate"] = None
    else:
        st.session_state["text_to_translate"] = st.text_area(
            label="Sentences to translate", label_visibility="collapsed"
        )


def _get_increment_size(num_sents) -> int:
    if cli_args().transl_batch_size >= num_sents:
        return 100
    else:
        return ceil(100 / (num_sents / cli_args().transl_batch_size))


def _translate():
    st.markdown("## Translations")
    download_info = st.empty()

    pbar = st.progress(0)
    sentences = [s.strip() for s in st.session_state["text_to_translate"].splitlines() if s.strip()]
    num_sentences = len(sentences)

    transl_info = st.info(f"Translating {num_sentences:,} sentence(s)...")

    increment = _get_increment_size(num_sentences)
    percent_done = 0
    all_translations = []

    transl_ct = st.empty()
    df = pd.DataFrame()
    for translations in st.session_state["translator"].batch_translate(
        sentences,
        batch_size=cli_args().transl_batch_size,
        max_length=cli_args().transl_max_length,
        num_beams=cli_args().transl_num_beams,
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
        pbar.progress(min(percent_done, 100), text=f"Translating: {min(percent_done, 100):.2f}%")

    pbar.empty()
    transl_info.success("Done translating!")

    xlsx_download_html = create_download_link(df, "translations.xlsx", "Download Excel")
    txt_download_html = create_download_link("\n".join(all_translations) + "\n", "translations.txt", "Download text")
    download_info.markdown(
        f"- <strong>{xlsx_download_html}</strong>: parallel source/MT sentences as Excel file;\n"
        f"- <strong>{txt_download_html}</strong>: only translations, as a plain text file.",
        unsafe_allow_html=True,
    )


def main():
    _init()
    _model_selection()
    _data_input()

    enabled = True
    msg = "Make sure that the following requirements are met:\n"
    if not ("translator" in st.session_state and st.session_state["translator"]):
        enabled = False
        msg += "- Translator not loaded\n"

    if not ("text_to_translate" in st.session_state and st.session_state["text_to_translate"]):
        enabled = False
        msg += "- Add text to translate\n"

    msg_container = st.empty()
    if not enabled:
        msg_container.warning(msg)
    else:
        msg_container.empty()
        if st.button("Translate"):
            _translate()


if __name__ == "__main__":
    main()
    # Call this to immediately disable CUDA if needed
    cli_args()
