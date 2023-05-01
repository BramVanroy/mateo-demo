import base64
import os
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import streamlit as st
from mateo_st.translator import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_SIZE,
    DEFAULT_NUM_BEAMS,
    TRANS_SIZE2MODEL,
)


def create_download_link(
    data: Union[str, pd.DataFrame], filename: str, link_text: str = "Download", df_groupby: Optional[str] = None
):
    if isinstance(data, pd.DataFrame):
        # Write the DataFrame to an in-memory bytes object
        bytes_io = BytesIO()
        with pd.ExcelWriter(bytes_io, "xlsxwriter") as writer:
            if df_groupby is not None:
                for groupname, groupdf in data.groupby(df_groupby):
                    groupdf = groupdf.drop(columns="metric").reset_index(drop=True)
                    groupdf.to_excel(writer, index=False, sheet_name=groupname)
            else:
                data.to_excel(writer, index=False)

        # Retrieve the bytes from the bytes object
        b64 = base64.b64encode(bytes_io.getvalue()).decode("utf-8")
        return f'<a download="{filename}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" title="Download">{link_text}</a>'
    elif isinstance(data, str):
        b64 = base64.b64encode(data.encode("utf-8")).decode("utf-8")
        return f'<a download="{filename}" href="data:file/txt;base64,{b64}" title="Download">{link_text}</a>'


@st.cache_data
def cli_args():
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        help="number of beams to  allow to generate translations with",
    )
    cparser.add_argument(
        "--transl_max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="maximal length to generate per sentence",
    )
    cparser.add_argument(
        "--eval_max_sys",
        type=int,
        default=4,
        help="max. number of systems to compare",
    )

    args = cparser.parse_args()

    # Disable CUDA for everything
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    return args


def load_css(name: str):
    pfcss = Path(__file__).parent.joinpath(f"css/{name}.css")
    st.markdown(f"<style>{read_file(pfcss)}</style>", unsafe_allow_html=True)


@st.cache_data(max_entries=64)
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
