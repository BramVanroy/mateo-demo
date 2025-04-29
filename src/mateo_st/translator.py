import os
from dataclasses import dataclass, field
from typing import Any, Union

import streamlit as st
import torch
from torch import nn, qint8
from torch.quantization import quantize_dynamic


DEFAULT_MODEL_SIZE = os.getenv("mateo_transl_model_size", "distilled-600M")
DEFAULT_BATCH_SIZE = os.getenv("mateo_batch_size", 1)
DEFAULT_MAX_LENGTH = os.getenv("mateo_max_length", 128)
DEFAULT_NUM_BEAMS = os.getenv("mateo_max_num_beams", 1)


@dataclass
class Translator:
    src_lang: str
    tgt_lang: str
    model_size: str = DEFAULT_MODEL_SIZE
    no_cuda: bool = True
    quantize: bool = True
    model: Any = field(default=None, init=False)
    tokenizer: Any = field(default=None, init=False)
    src_lang_key: str = field(default=None, init=False)
    tgt_lang_key: str = field(default=None, init=False)

    def __post_init__(self):
        try:
            self.model_name = TRANS_SIZE2MODEL[self.model_size]
        except KeyError:
            raise KeyError(f"Model size '{self.model_size}' not recognized or supported")

        if not torch.cuda.is_available():
            self.no_cuda = True

        self.model, self.tokenizer, self.no_cuda = init_model(self.model_name, self.no_cuda, self.quantize)
        self.set_src_lang(self.src_lang)
        self.set_tgt_lang(self.tgt_lang)
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

    @torch.inference_mode()
    def translate(self, encoded, max_length: int = DEFAULT_MAX_LENGTH, num_beams: int = DEFAULT_NUM_BEAMS):
        return self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang_key),
            max_length=max_length,
            num_beams=num_beams,
        )

    def batch_translate(
        self,
        sentences: Union[str, list[str]],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        for batch in batchify(sentences, batch_size):
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True)
            if not self.no_cuda:
                encoded = encoded.to("cuda")

            try:
                generated_tokens = self.translate(encoded, max_length=max_length, num_beams=num_beams)
            except RuntimeError:
                # Out-of-memory; switch to CPU
                self.no_cuda = True
                self.model = self.model.to("cpu")
                encoded = encoded.to("cpu")
                generated_tokens = self.translate(encoded, max_length=max_length, num_beams=num_beams)

            yield self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def batchify(sentences: list[str], batch_size: int):
    """Yields batches of size 'batch_size' from the given list of sentences"""
    num_sents = len(sentences)
    for idx in range(0, num_sents, batch_size):
        yield sentences[idx : idx + batch_size]


def _validate_cached_resources(cached_resource: tuple):
    """Validate the cached resources. Sometimes a model/tokenizer can be None, e.g. when the HF hub is down and the
    model cannot be downloaded. That will lead to a cached None value. We do not want new users to also
    have to use the cached None value, so we validate it here. If it was None, we return False so that the cache
    will be discarded.

    :param cached_resource: a tuple of (model, tokenizer, no_cuda)
    :return: False if the model or tokenizer is None (not valid) otherwise True
    """
    if any(item is None for item in cached_resource[:-1]):
        return False
    else:
        return True


@st.cache_resource(show_spinner=False, max_entries=1, validate=_validate_cached_resources)
def init_model(model_name: str, no_cuda: bool = True, quantize: bool = True):
    # We defer loading of transformers and optimum because _sometimes_ there are
    # import errors triggered if we put them at the top. Don't know why...
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available() and not no_cuda:
        try:
            model = model.to("cuda")
        except RuntimeError:
            # In case of out-of-memory on the GPU when moving model to GPU,
            # just stay on CPU and disable CUDA
            no_cuda = True
    elif quantize:  # Quantization not supported on CUDA
        model = quantize_dynamic(model, {nn.Linear, nn.Dropout, nn.LayerNorm}, dtype=qint8)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, no_cuda


def update_translator_lang(side: str):
    st.session_state[f"{side}_lang_key"] = TRANS_LANG2KEY[st.session_state[f"{side}_lang"]]
    if "translator" in st.session_state and st.session_state["translator"]:
        if side == "src":
            st.session_state["translator"].set_src_lang(st.session_state["src_lang"])
        else:
            st.session_state["translator"].set_tgt_lang(st.session_state["tgt_lang"])


TRANS_LANG2KEY = {
    "Acehnese (Arabic script)": "ace_Arab",
    "Acehnese (Latin script)": "ace_Latn",
    "Mesopotamian Arabic": "acm_Arab",
    "Ta’izzi-Adeni Arabic": "acq_Arab",
    "Tunisian Arabic": "aeb_Arab",
    "Afrikaans": "afr_Latn",
    "South Levantine Arabic": "ajp_Arab",
    "Akan": "aka_Latn",
    "Amharic": "amh_Ethi",
    "North Levantine Arabic": "apc_Arab",
    "Modern Standard Arabic": "arb_Arab",
    "Modern Standard Arabic (Romanized)": "arb_Latn",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Central Aymara": "ayr_Latn",
    "South Azerbaijani": "azb_Arab",
    "North Azerbaijani": "azj_Latn",
    "Bashkir": "bak_Cyrl",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Banjar (Arabic script)": "bjn_Arab",
    "Banjar (Latin script)": "bjn_Latn",
    "Standard Tibetan": "bod_Tibt",
    "Bosnian": "bos_Latn",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Czech": "ces_Latn",
    "Chokwe": "cjk_Latn",
    "Central Kurdish": "ckb_Arab",
    "Crimean Tatar": "crh_Latn",
    "Welsh": "cym_Latn",
    "Danish": "dan_Latn",
    "German": "deu_Latn",
    "Southwestern Dinka": "dik_Latn",
    "Dyula": "dyu_Latn",
    "Dzongkha": "dzo_Tibt",
    "Greek": "ell_Grek",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Basque": "eus_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Nigerian Fulfulde": "fuv_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Irish": "gle_Latn",
    "Galician": "glg_Latn",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian Creole": "hat_Latn",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Croatian": "hrv_Latn",
    "Hungarian": "hun_Latn",
    "Armenian": "hye_Armn",
    "Igbo": "ibo_Latn",
    "Ilocano": "ilo_Latn",
    "Indonesian": "ind_Latn",
    "Icelandic": "isl_Latn",
    "Italian": "ita_Latn",
    "Javanese": "jav_Latn",
    "Japanese": "jpn_Jpan",
    "Kabyle": "kab_Latn",
    "Jingpho": "kac_Latn",
    "Kamba": "kam_Latn",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic script)": "kas_Arab",
    "Kashmiri (Devanagari script)": "kas_Deva",
    "Georgian": "kat_Geor",
    "Central Kanuri (Arabic script)": "knc_Arab",
    "Central Kanuri (Latin script)": "knc_Latn",
    "Kazakh": "kaz_Cyrl",
    "Kabiyè": "kbp_Latn",
    "Kabuverdianu": "kea_Latn",
    "Khmer": "khm_Khmr",
    "Kikuyu": "kik_Latn",
    "Kinyarwanda": "kin_Latn",
    "Kyrgyz": "kir_Cyrl",
    "Kimbundu": "kmb_Latn",
    "Northern Kurdish": "kmr_Latn",
    "Kikongo": "kon_Latn",
    "Korean": "kor_Hang",
    "Lao": "lao_Laoo",
    "Ligurian": "lij_Latn",
    "Limburgish": "lim_Latn",
    "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn",
    "Lombard": "lmo_Latn",
    "Latgalian": "ltg_Latn",
    "Luxembourgish": "ltz_Latn",
    "Luba-Kasai": "lua_Latn",
    "Ganda": "lug_Latn",
    "Luo": "luo_Latn",
    "Mizo": "lus_Latn",
    "Standard Latvian": "lvs_Latn",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Minangkabau (Arabic script)": "min_Arab",
    "Minangkabau (Latin script)": "min_Latn",
    "Macedonian": "mkd_Cyrl",
    "Plateau Malagasy": "plt_Latn",
    "Maltese": "mlt_Latn",
    "Meitei (Bengali script)": "mni_Beng",
    "Halh Mongolian": "khk_Cyrl",
    "Mossi": "mos_Latn",
    "Maori": "mri_Latn",
    "Burmese": "mya_Mymr",
    "Dutch": "nld_Latn",
    "Norwegian Nynorsk": "nno_Latn",
    "Norwegian Bokmål": "nob_Latn",
    "Nepali": "npi_Deva",
    "Northern Sotho": "nso_Latn",
    "Nuer": "nus_Latn",
    "Nyanja": "nya_Latn",
    "Occitan": "oci_Latn",
    "West Central Oromo": "gaz_Latn",
    "Odia": "ory_Orya",
    "Pangasinan": "pag_Latn",
    "Eastern Panjabi": "pan_Guru",
    "Papiamento": "pap_Latn",
    "Western Persian": "pes_Arab",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Dari": "prs_Arab",
    "Southern Pashto": "pbt_Arab",
    "Ayacucho Quechua": "quy_Latn",
    "Romanian": "ron_Latn",
    "Rundi": "run_Latn",
    "Russian": "rus_Cyrl",
    "Sango": "sag_Latn",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sicilian": "scn_Latn",
    "Shan": "shn_Mymr",
    "Sinhala": "sin_Sinh",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Samoan": "smo_Latn",
    "Shona": "sna_Latn",
    "Sindhi": "snd_Arab",
    "Somali": "som_Latn",
    "Southern Sotho": "sot_Latn",
    "Spanish": "spa_Latn",
    "Tosk Albanian": "als_Latn",
    "Sardinian": "srd_Latn",
    "Serbian": "srp_Cyrl",
    "Swati": "ssw_Latn",
    "Sundanese": "sun_Latn",
    "Swedish": "swe_Latn",
    "Swahili": "swh_Latn",
    "Silesian": "szl_Latn",
    "Tamil": "tam_Taml",
    "Tatar": "tat_Cyrl",
    "Telugu": "tel_Telu",
    "Tajik": "tgk_Cyrl",
    "Tagalog": "tgl_Latn",
    "Thai": "tha_Thai",
    "Tigrinya": "tir_Ethi",
    "Tamasheq (Latin script)": "taq_Latn",
    "Tamasheq (Tifinagh script)": "taq_Tfng",
    "Tok Pisin": "tpi_Latn",
    "Tswana": "tsn_Latn",
    "Tsonga": "tso_Latn",
    "Turkmen": "tuk_Latn",
    "Tumbuka": "tum_Latn",
    "Turkish": "tur_Latn",
    "Twi": "twi_Latn",
    "Central Atlas Tamazight": "tzm_Tfng",
    "Uyghur": "uig_Arab",
    "Ukrainian": "ukr_Cyrl",
    "Umbundu": "umb_Latn",
    "Urdu": "urd_Arab",
    "Northern Uzbek": "uzn_Latn",
    "Venetian": "vec_Latn",
    "Vietnamese": "vie_Latn",
    "Waray": "war_Latn",
    "Wolof": "wol_Latn",
    "Xhosa": "xho_Latn",
    "Eastern Yiddish": "ydd_Hebr",
    "Yoruba": "yor_Latn",
    "Yue Chinese": "yue_Hant",
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Standard Malay": "zsm_Latn",
    "Zulu": "zul_Latn",
}

TRANS_KEY2LANG = {v: k for k, v in TRANS_LANG2KEY.items()}

TRANS_SIZE2MODEL = {
    "distilled-600M": "facebook/nllb-200-distilled-600M",
    "1.3B": "facebook/nllb-200-1.3B",
    "distilled-1.3B": "facebook/nllb-200-distilled-1.3B",
    "3.3B": "facebook/nllb-200-3.3B",
}
