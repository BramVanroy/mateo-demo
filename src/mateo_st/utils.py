import base64
import json
import logging
import os
from argparse import Namespace
from io import BytesIO, StringIO
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
import torch

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

    defaults = {
        "use_cuda": False,
        "transl_batch_size": DEFAULT_BATCH_SIZE,
        "transl_no_quantize": False,
        "transl_model_size": DEFAULT_MODEL_SIZE,
        "transl_num_beams": DEFAULT_NUM_BEAMS,
        "transl_max_length": DEFAULT_MAX_LENGTH,
        "eval_max_sys": 4,
        "demo_mode": False,

    }
    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument("--use_cuda", default=False, action="store_true", help="whether to use CUDA. Only affects the translation model")
    cparser.add_argument(
        "--demo_mode",
        action="store_true",
        default=False,
        help="when demo mode is enabled, only a limited range of neural check-points are available. So all metrics are"
        " available but not all of the checkpoints.",
    )

    args = cparser.parse_args()

    # Options specified in the JSON config overwrite CLI args
    args = Namespace(**{**defaults, **vars(args)})
    if not torch.cuda.is_available():
        args.use_cuda = False
        logging.warning("CUDA is not available on this system. Disabling it.")

    # Disable CUDA for everything
    if not args.use_cuda:
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


def get_uploaded_file_as_strio(uploaded_file: BytesIO) -> StringIO | None:
    """Decode a given file (in-memory, BytesIO object) and throw a ST error message when an error occurs. Likely
    in case of encoding issues.

    :param uploaded_file: the in-memory bytes of the uploaded file
    :return: the utf-8 encoded file contents or None
    """
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    except UnicodeDecodeError:
        st.error(
            "Could not decode parts of your file because it is in an unexpected encoding. MATEO expects"
            " your file to be encoded as UTF-8!"
        )
    except Exception as exc:
        st.error(
            f"""Something went wrong when uploading your data. Try again later and if the error persists, submit an issue on [Github](https://github.com/BramVanroy/mateo-demo/issues)! Include this error:

    {exc}"""
        )
    else:
        return stringio


def print_citation_info(
    expander_text: Optional[str] = "✒️ If you use MATEO for your work, please **cite it** accordingly.",
):
    """Print in streamlit the MATEO citation information, optionally inside an expander element
    :param expander_text: text for expander. If None or empty string, will not use the expander
    """
    apa_bibtex = """> Vanroy, B., Tezcan, A., & Macken, L. (2023). [MATEO: MAchine Translation Evaluation Online](https://aclanthology.org/2023.eamt-1.52/). In M. Nurminen, J. Brenner, M. Koponen, S. Latomaa, M. Mikhailov, F. Schierl, … H. Moniz (Eds.), _Proceedings of the 24th Annual Conference of the European Association for Machine Translation_ (pp. 499–500). Tampere, Finland: European Association for Machine Translation (EAMT).

```bibtex
@inproceedings{vanroy-etal-2023-mateo,
    title = "{MATEO}: {MA}chine {T}ranslation {E}valuation {O}nline",
    author = "Vanroy, Bram  and
      Tezcan, Arda  and
      Macken, Lieve",
    booktitle = "Proceedings of the 24th Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2023",
    address = "Tampere, Finland",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2023.eamt-1.52",
    pages = "499--500",
}
```
"""

    if expander_text:
        with st.expander(expander_text):
            st.markdown(apa_bibtex)
    else:
        st.markdown(apa_bibtex)
