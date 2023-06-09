import base64
import os
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
from mateo_st.translator import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_SIZE,
    DEFAULT_NUM_BEAMS,
    TRANS_SIZE2MODEL,
)
from matplotlib.image import imread


def create_download_link(
    data: Union[str, pd.DataFrame], filename: str, link_text: str = "Download", df_groupby: Optional[str] = None
) -> str:
    """Given a string or dataframe, turn it into an in-memory object that can be downloaded
    :param data: data to include
    :param filename: stem of the filename that we create in-memory
    :param link_text: text of the hyperlink that will be created
    :param df_groupby: whether to create separate sheets for a given "df_groupby" column. Only used if 'data' is
     a DataFrame
    :return: an HTML <a> link that contains the download item as base64 encoded content
    """
    if isinstance(data, pd.DataFrame):
        # Write the DataFrame to an in-memory bytes object
        bytes_io = BytesIO()
        with pd.ExcelWriter(bytes_io, "xlsxwriter") as writer:
            # df_groupby is used especially in the sentence-level Excel sheet where we want to create
            # separate sheets per metric
            if df_groupby is not None:
                for groupname, groupdf in data.groupby(df_groupby):
                    groupdf = groupdf.drop(columns=df_groupby).reset_index(drop=True)
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
    cparser.add_argument(
        "--demo_mode",
        action="store_true",
        default=False,
        help="when demo mode is enabled, only a limited range of neural check-points are available. So all metrics are"
        " available but not all of the checkpoints.",
    )

    args = cparser.parse_args()

    # Disable CUDA for everything
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    return args


def load_css(stem: str):
    """Load a given CSS file in streamlit. The given "name".css will be looked for in the css directory.
    :param stem: filename without extension to load
    """
    pfcss = Path(__file__).parent.joinpath(f"css/{stem}.css")
    st.markdown(f"<style>{read_local_file(pfcss)}</style>", unsafe_allow_html=True)


@st.cache_data(max_entries=64)
def read_local_file(fin: Union[str, PathLike]) -> str:
    return Path(fin).read_text(encoding="utf-8")


def get_local_img(name: str) -> np.ndarray:
    """Get the full path to a local image
    :param name: filename with extension
    """
    return imread(Path(__file__).parent.joinpath(f"img/{name}").resolve())


def isfloat(item: str) -> bool:
    """Check whether the given item can in fact be a float
    :param item: the item to check
    :return: True if it could be a float, False if not
    """
    try:
        float(item)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(item: str) -> bool:
    """Check whether the given item can in fact be an integer
    :param item: the item to check
    :return: True if it could be an int, False if not
    """
    try:
        item_float = float(item)
        item_int = int(item)
    except (TypeError, ValueError):
        return False
    else:
        return item_float == item_int


def build_signature(paired_bs_n: int, seed: int, library_version: str, metric_options: dict) -> str:
    """Builds a signature for a metric that does not support it out of the box (SacreBLEU does)
    :param paired_bs_n: number of resamples in bootstrap resampling
    :param seed: random seed used in bootstrap resampling
    :param library_version: version of the library that this metric belongs to
    :param metric_options: additional options that were specified in this metric
    :return:
    """
    # Sort options to ensure determinism in abbreviations
    metric_options = {prop: metric_options[prop] for prop in sorted(metric_options.keys())}

    sig = f"nrefs:1|bs:{paired_bs_n}|seed:{seed}"

    abbrs = set()
    # Iteratively add all the options. As keys in the signature, just use the first letter
    # of the property. If it already exists, use the first two letters, etc.
    for prop, value in metric_options.items():
        if value == "" or value is None:
            continue
        idx = 1
        abbr = prop[:idx]
        while abbr in abbrs:
            idx += 1
            abbr = prop[:idx]
        abbrs.add(abbr)

        # Convert bools to yes/no
        if isinstance(value, bool):
            value = "yes" if value is True else "no"

        sig += f"|{abbr}:{value}"

    sig += f"|version:{library_version}"

    return sig
