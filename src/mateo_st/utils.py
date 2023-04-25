import base64
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Union

import pandas as pd
import streamlit as st
from mateo_st.translator import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_SIZE,
    DEFAULT_NUM_BEAMS,
    TRANS_LANG2KEY,
    TRANS_SIZE2MODEL,
)


def update_translator_lang(side: str):
    st.session_state[f"{side}_lang_key"] = TRANS_LANG2KEY[st.session_state[f"{side}_lang"]]
    if "translator" in st.session_state and st.session_state["translator"]:
        if side == "src":
            st.session_state["translator"].set_src_lang(st.session_state["src_lang"])
        else:
            st.session_state["translator"].set_tgt_lang(st.session_state["tgt_lang"])


COLORS_PLOTLY = {
    "default": ["#94bf89", "#BF8480", "#6A5E73"],
    # Add other styles if wanted
}


@st.cache_data
def cli_args():
    import argparse

    cparser = argparse.ArgumentParser()
    cparser.add_argument("--no_cuda", action="store_true", help="whether to disable CUDA for all tasks")
    cparser.add_argument("--transl_no_cuda", action="store_true", help="whether to disable CUDA for translation")

    cparser.add_argument(
        "--transl_batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="batch size for translating"
    )
    cparser.add_argument(
        "--transl_no_quantize",
        action="store_true",
        help="whether to disable CUDA torch quantization of the translation model. Quantization makes the model smaller"
        " and faster but may result in lower quality. This option will disable quantization.",
    )
    cparser.add_argument(
        "--transl_model_size",
        choices=list(TRANS_SIZE2MODEL.keys()),
        default=DEFAULT_MODEL_SIZE,
        help="translation model size to use",
    )
    cparser.add_argument(
        "--transl_num_beams",
        type=int,
        default=DEFAULT_NUM_BEAMS,
        help="number of beams to generate translations with",
    )
    cparser.add_argument(
        "--transl_max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="maximal length to generate per sentence",
    )

    return cparser.parse_args()


def set_general_session_keys():
    # CUDA
    if "no_cuda" not in st.session_state:
        st.session_state["no_cuda"] = cli_args().no_cuda

    if "transl_no_cuda" not in st.session_state:
        st.session_state["transl_no_cuda"] = cli_args().transl_no_cuda

    # TRANSLATION
    if "transl_batch_size" not in st.session_state:
        st.session_state["transl_batch_size"] = cli_args().transl_batch_size

    if "transl_model_size" not in st.session_state:
        st.session_state["transl_model_size"] = cli_args().transl_model_size

    if "transl_num_beams" not in st.session_state:
        st.session_state["transl_num_beams"] = cli_args().transl_num_beams

    if "transl_max_length" not in st.session_state:
        st.session_state["transl_max_length"] = cli_args().transl_max_length


def create_download_link(data: Union[str, pd.DataFrame], filename: str, link_text: str = "Download"):
    if isinstance(data, pd.DataFrame):
        # Write the DataFrame to an in-memory bytes object
        bytes_io = BytesIO()
        with pd.ExcelWriter(bytes_io, "xlsxwriter") as writer:
            data.to_excel(writer, index=False)

        # Retrieve the bytes from the bytes object
        b64 = base64.b64encode(bytes_io.getvalue()).decode("utf-8")
        return (
            f'<a download="{filename}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml'
            f'.sheet;base64,{b64}" title="Download">{link_text}</a>'
        )
    elif isinstance(data, str):
        b64 = base64.b64encode(data.encode("utf-8")).decode("utf-8")
        return f'<a download="{filename}" href="data:file/txt;base64,{b64}" title="Download">{link_text}</a>'


def load_css(name: str):
    pfcss = Path(__file__).parent.joinpath(f"css/{name}.css")
    st.markdown(f"<style>{read_file(pfcss)}</style>", unsafe_allow_html=True)


@st.cache_data
def read_file(fin: Union[str, PathLike]):
    return Path(fin).read_text(encoding="utf-8")


def isfloat(item: str):
    try:
        float(item)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(item: str):
    try:
        item_float = float(item)
        item_int = int(item)
    except (TypeError, ValueError):
        return False
    else:
        return item_float == item_int
