from dataclasses import dataclass, field
from typing import List, Union, Dict

import streamlit as st
import torch
from torch.quantization import quantize_dynamic
from torch import nn, qint8, Tensor
from torch.nn import Parameter
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class Translator:
    src_lang: str
    tgt_lang: str
    model_size: str = "418M"
    no_cuda: bool = False
    max_length: int = 256
    model: M2M100ForConditionalGeneration = field(default=None, init=False)
    tokenizer: M2M100Tokenizer = field(default=None, init=False)
    src_lang_key: str = field(default=None, init=False)
    tgt_lang_key: str = field(default=None, init=False)

    def __post_init__(self):
        try:
            self.model_name = TRANS_SIZE2MODEL[self.model_size]
        except KeyError:
            raise KeyError(f"Model size '{self.model_size}' not recognized or supported")

        if not torch.cuda.is_available():
            self.no_cuda = True

        self.model, self.tokenizer = init_model(self.model_name)
        self.set_src_lang(self.src_lang)
        self.set_tgt_lang(self.tgt_lang)

        if not self.no_cuda:
            try:
                self.model = self.model.to("cuda")
            except RuntimeError:
                # Out-of-memory
                self.no_cuda = True

        self.model.eval()

    def set_src_lang(self, src_lang):
        self.src_lang = src_lang
        try:
            self.src_lang_key = TRANS_LANG2KEY[self.src_lang]
        except KeyError:
            raise KeyError(f"Source language '{self.src_lang}' not recognized or supported")
        self.tokenizer.src_lang = self.src_lang_key

    def set_tgt_lang(self, tgt_lang):
        self.tgt_lang = tgt_lang
        try:
            self.tgt_lang_key = TRANS_LANG2KEY[self.tgt_lang]
        except KeyError:
            raise KeyError(f"Target language '{self.tgt_lang}' not recognized or supported")
        self.tokenizer.tgt_lang = self.tgt_lang_key


def get_translator_hash(translator: Translator):
    return (
        translator.src_lang,
        translator.tgt_lang,
        translator.model_size,
        translator.model_name,
        translator.max_length,
    )


def batch_translate(translator, sentences: Union[str, List[str]], batch_size: int = 4):
    if isinstance(sentences, str):
        sentences = [sentences]

    for batch in batchify(sentences, batch_size):
        encoded = translator.tokenizer(batch, return_tensors="pt", padding=True)
        if not translator.no_cuda:
            encoded = encoded.to("cuda")

        try:
            generated_tokens = _translate(translator, encoded)
        except RuntimeError:
            # Out-of-memory; switch to CPU
            translator.no_cuda = True
            translator.model = translator.model.to("cpu")
            encoded = encoded.to("cpu")
            generated_tokens = _translate(translator, encoded)

        yield translator.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


# @st.cache(
#     allow_output_mutation=True,
#     suppress_st_warning=False,
#     show_spinner=False,
#     hash_funcs={Translator: get_translator_hash,
#                 PreTrainedModel: lambda model: model.name_or_path,
#                 PreTrainedTokenizer: lambda tokenizer: tokenizer.name_or_path,
#                 Parameter: lambda parameter: parameter.data,
#                 Tensor: lambda tensor: tensor.cpu()},
# )
def _translate(translator, encoded: Dict[str, int]):
    return translator.model.generate(
        **encoded,
        forced_bos_token_id=translator.tokenizer.get_lang_id(translator.tgt_lang_key),
        max_length=translator.max_length,
        num_beams=5,
    )

def batchify(sentences: List[str], batch_size: int):
    """Yields batches of size 'batch_size' from the given list of sentences"""
    num_sents = len(sentences)
    for idx in range(0, num_sents, batch_size):
        yield sentences[idx:idx + batch_size]


@st.cache(allow_output_mutation=True, suppress_st_warning=False, show_spinner=False)
def init_model(model_name: str):
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)

    return model, tokenizer


TRANS_LANG2KEY = {
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Asturian": "ast",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Breton": "br",
    "Bulgarian": "bg",
    "Burmese": "my",
    "Catalan; Valencian": "ca",
    "Cebuano": "ceb",
    "Central Khmer": "km",
    "Chinese": "zh",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch; Flemish": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Fulah": "ff",
    "Gaelic; Scottish Gaelic": "gd",
    "Galician": "gl",
    "Ganda": "lg",
    "Georgian": "ka",
    "German": "de",
    "Greeek": "el",
    "Gujarati": "gu",
    "Haitian; Haitian Creole": "ht",
    "Hausa": "ha",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Igbo": "ig",
    "Iloko": "ilo",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Korean": "ko",
    "Lao": "lo",
    "Latvian": "lv",
    "Lingala": "ln",
    "Lithuanian": "lt",
    "Luxembourgish; Letzeburgesch": "lb",
    "Macedonian": "mk",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Mongolian": "mn",
    "Nepali": "ne",
    "Northern Sotho": "ns",
    "Norwegian": "no",
    "Occitan": "oc",
    "Oriya": "or",
    "Panjabi; Punjabi": "pa",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Pushto; Pashto": "ps",
    "Romanian; Moldavian; Moldovan": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Sindhi": "sd",
    "Sinhala; Sinhalese": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swati": "ss",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tamil": "ta",
    "Thai": "th",
    "Tswana": "tn",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Western Frisian": "fy",
    "Wolof": "wo",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Zulu": "zu",
}

TRANS_KEY2LANG = {v: k for k, v in TRANS_LANG2KEY.items()}

TRANS_SIZE2MODEL = {
    "418M": "facebook/m2m100_418M",
    "1.2B": "facebook/m2m100_1.2B",
    # "12B": "facebook/m2m100-12B-avg-5-ckpt",
}
