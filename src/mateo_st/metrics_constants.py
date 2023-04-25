from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, Type

import bert_score
import comet
import sacrebleu
from bert_score import lang2model as bertscore_lang2model
from sacrebleu import BLEU
from sacrebleu.metrics.bleu import _TOKENIZERS as SBTOKENIZERS


@dataclass
class MetricOption:
    name: str
    description: str
    default: Any
    choices: Optional[Tuple] = None
    types: Optional[Tuple[Type, ...]] = None
    allow_empty_str: bool = False

    def __post_init__(self):
        if not self.choices and not self.types:
            raise ValueError(f"{self.name} needs at least one of 'choices' or 'types'")

        if self.allow_empty_str and any(t is not int and t is not float for t in self.types):
            raise ValueError(f"{self.name}: 'allow_empty_str' only makes sense if the input types are int or float")

        if self.choices and self.default not in self.choices:
            raise ValueError(f"{self.name}: the default option ('{self.default}') must be in 'choices' ('{', '.join(self.choices)}')")


@dataclass
class MetricMeta:
    name: str
    metric_class: Literal["baseline", "neural"]
    full_name: str
    description_html: str
    paper_url: str
    implementation_html: str
    evaluate_name: str
    is_default_selected: bool = False
    higher_better: bool = True
    version: Optional[str] = None
    options: Optional[Tuple[MetricOption, ...]] = None
    requires_source: bool = False


METRICS_META = {
    "bertscore": MetricMeta(
        name="BERTScore",
        metric_class="neural",
        full_name="",
        description_html="BERTScore is an automatic evaluation metric for text generation that computes similarity scores"
        " for each token in the machine translation with each token in the reference translation. These"
        " similarity scores are based on contextual embeddings retrieved from a"
        " <a href='https://aclanthology.org/N19-1423/' title='BERT paper'>BERT</a>"
        " language model.</p>",
        paper_url="https://openreview.net/forum?id=SkeHuCVFDr",
        implementation_html="<p><a href='https://github.com/Tiiiger/bert_score' title='BertScore GitHub'"
        ">BERTScore</a>. <a href='https://github.com/Tiiiger/bert_score#default-behavior'"
        " title='Default BERTScore behavior'>The underlying model</a> depends on the target language"
        " of your example.</p>",
        evaluate_name="bertscore",
        version=bert_score.__version__,
        options=(
            MetricOption(
                name="lang",
                description="Language of the translations. This is an optional shortcut, used to select a good default"
                            f" model for your language"
                            f" ({', '.join([': '.join(langmodel) for langmodel in bertscore_lang2model.items()])}, and"
                            f" bert-base-multilingual-cased for 'other').\n"
                            " Alternatively, choose a model from the 'model_type' option.\n⚠️ 'model_type' has"
                            " precedence over 'lang' so make sure to set 'model_type' to '' when selecting a 'lang'!",
                default="other",
                choices=("other", ) + tuple(bertscore_lang2model.keys()),
            ),
            MetricOption(
                name="model_type",
                description="Model type to use. For demo purposes, we have limited the options to a couple of"
                            " high-performing base and large models",
                default="",
                # Not all models because we want to save compute
                # All options here: https://github.com/Tiiiger/bert_score/blob/dbcf6db37e8bd6ff68446f06b0ba5d0763b62d20/bert_score/utils.py#L40
                choices=("",
                         "bert-base-multilingual-cased",
                         "roberta-base",
                         "roberta-large",
                         "bert-base-chinese",
                         "dbmdz/bert-base-turkish-cased",
                         "allenai/scibert_scivocab_uncased",
                         "facebook/bart-base",
                         "princeton-nlp/sup-simcse-bert-base-uncased",
                         "microsoft/deberta-v3-base",
                         "microsoft/deberta-v3-large",
                         "microsoft/mdeberta-v3-base"),
            ),
            MetricOption(
                name="num_layers",
                description="This layer's representation will be used. If empty, defaults to the best layer as tuned"
                            " on WMT16",
                default="",
                types=(int,),
                allow_empty_str=True,
            )
            # Not adding other options such as rescale_with_baseline or idf, because those require extra corpus input
            # to calculate baseline/idf scores on
        ),
    ),
    "bleu": MetricMeta(
        name="BLEU",
        metric_class="baseline",
        full_name="BiLingual Evaluation Understudy",
        description_html="<p>BLEU is perhaps the most well-known MT evaluation metric. It relies on how well a machine"
        " translation's n-grams correspond to those of a reference translation. Despite its popularity,"
        " it has also been criticized for its shortcomings such as a lack of sufficiently incorporating"
        " recall (e.g., <a href='https://aclanthology.org/E06-1032'"
        " title='Callison-Burch, Osborne, Koehn criticism on BLEU'>[1]</a>).</p>",
        paper_url="https://aclanthology.org/P02-1040/",
        implementation_html="<p><a href='https://github.com/mjpost/sacrebleu' title='SacreBLEU GitHub'>SacreBLEU</a></p>",
        is_default_selected=True,
        evaluate_name="sacrebleu",
        version=sacrebleu.__version__,
        options=(
            MetricOption(
                name="smooth_method",
                description="Smoothing method to use",
                default="exp",
                choices=("floor", "add-k", "exp", "none"),
            ),
            MetricOption(
                name="smooth_value",
                description="Smoothing value for `floor` and `add-k` methods. An empty value falls back to the default value",
                default="",
                types=(float, int),
                allow_empty_str=True,
            ),
            MetricOption(name="lowercase", description="Whether to lowercase the data", default=False, types=(bool,)),
            MetricOption(
                name="tokenize",
                description="Tokenizer to use",
                default=BLEU.TOKENIZER_DEFAULT,
                choices=tuple(SBTOKENIZERS.keys()),
            ),
        ),
    ),
    "bleurt": MetricMeta(
        name="BLEURT",
        metric_class="neural",
        full_name="Bilingual Evaluation Understudy with Representations from Transformers",
        description_html="BLEURT utilizes large language models <a href='https://aclanthology.org/N19-1423/'"
        " title='BERT paper'>BERT</a> and"
        " <a href='https://openreview.net/forum?id=xpFFI_NtgpW' title='RemBERT paper'"
        ">RemBERT</a> to compare a machine translation to a reference translation."
        " In their work, they highlight that most of their improvements are thanks to using synthetic"
        " data and pretraining on many different tasks such as back translation, entailment, and"
        " predicting existing MT metrics such as BLEU and BERTScore.</p>",
        paper_url="https://aclanthology.org/2020.acl-main.704/",
        implementation_html="<p><a href='https://github.com/google-research/bleurt' title='BLEURT GitHub'>BLEURT</a>."
        " We use the <code>BLEURT-20</code> checkpoint in this demo.</p>",
        evaluate_name="bleurt",
        version="commit cebe7e6",
        options=(
            MetricOption(
                name="checkpoint",
                description="BLEURT trained checkpoint to use",
                default="BLEURT-20",
                choices=("bleurt-tiny-128", "bleurt-tiny-512", "bleurt-base-128", "bleurt-base-512", "bleurt-large-128",
                         "bleurt-large-512", "BLEURT-20-D3", "BLEURT-20-D6", "BLEURT-20-D12", "BLEURT-20"),
            ),
        ),
    ),
    "chrf": MetricMeta(
        name="ChrF",
        metric_class="baseline",
        full_name="Character F-score",
        description_html="<p>ChrF uses the F-score statistic for character n-gram matches. It therefore focuses on"
        " characters rather than words. It has shown to be a strong baseline, often outperforming other"
        " baselines like BLEU or TER. By default, up to character 6-grams are considered.</p>",
        paper_url="https://aclanthology.org/W15-3049",
        implementation_html="<p><a href='https://github.com/mjpost/sacrebleu' title='SacreBLEU GitHub'>SacreBLEU</a></p>",
        is_default_selected=True,
        evaluate_name="chrf",
        version=sacrebleu.__version__,
        options=(),
    ),
    "comet": MetricMeta(
        name="COMET",
        metric_class="neural",
        full_name="Crosslingual Optimized Metric for Evaluation of Translation",
        description_html="COMET is similar to other neural approaches in that it also finetunes existing language models, "
        " in their case"
        " <a href='https://aclanthology.org/2020.acl-main.747/' title='XLM-R paper'>XLM-R</a>. What"
        " makes COMET different, however, is that it can also consider the source text as part of the"
        " input rather than only comparing a machine translation to a reference translation.</p>",
        paper_url="https://aclanthology.org/2020.emnlp-main.213/",
        implementation_html="<p><a href='https://github.com/Unbabel/COMET' title='COMET GitHub'>COMET</a>."
        " We use the default <code>wmt20-comet-da</code> checkpoint in this demo.</p>",
        is_default_selected=True,
        evaluate_name="comet",
        version=comet.__version__,
        options=(),
        requires_source=True,
    ),
    "ter": MetricMeta(
        name="TER",
        metric_class="baseline",
        full_name="Translation Edit Rate",
        description_html="TER is an intuitive baseline metric that formulates MT quality as the number of edits required"
        " to change an MT translation so that it exactly matches a reference translation. The possible"
        " edit operations are insertions, deletions, substitutions of words as well as word sequence"
        " ('phrase') shifts.</p>",
        paper_url="https://aclanthology.org/2006.amta-papers.25/",
        implementation_html="<p><a href='https://github.com/mjpost/sacrebleu' title='SacreBLEU GitHub'>SacreBLEU</a></p>",
        is_default_selected=True,
        higher_better=False,
        evaluate_name="ter",
        version=sacrebleu.__version__,
        options=(),
    ),
}

DEFAULT_METRICS = {name for name, meta in METRICS_META.items() if meta.is_default_selected}
BASELINE_METRICS = {name for name, meta in METRICS_META.items() if meta.metric_class == "baseline"}


SUPPORTED_LANGS = {
    "bertscore": {
        "Afrikaans": "af",
        "Albanian": "sq",
        "Arabic": "ar",
        "Aragonese": "an",
        "Armenian": "hy",
        "Asturian": "ast",
        "Azerbaijani": "az",
        "Bangla": "bn",
        "Bashkir": "ba",
        "Basque": "eu",
        "Bavarian": "bar",
        "Belarusian": "be",
        "Bishnupriya Manipuri": "bpy",
        "Bosnian": "bs",
        "Breton": "br",
        "Bulgarian": "bg",
        "Burmese": "my",
        "Catalan; Valencian": "ca",
        "Cebuano": "ceb",
        "Chechen": "ce",
        "Chinese": "zh",
        "Chuvash": "cv",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dutch; Flemish": "nl",
        "English": "en",
        "Estonian": "et",
        "Filipino": "tl",
        "Finnish": "fi",
        "French": "fr",
        "Galician": "gl",
        "Georgian": "ka",
        "German": "de",
        "Greeek": "el",
        "Gujarati": "gu",
        "Haitian; Haitian Creole": "ht",
        "Hebrew": "he",
        "Hindi": "hi",
        "Hungarian": "hu",
        "Icelandic": "is",
        "Ido": "io",
        "Indonesian": "id",
        "Irish": "ga",
        "Italian": "it",
        "Japanese": "ja",
        "Javanese": "jv",
        "Kannada": "kn",
        "Kazakh": "kk",
        "Korean": "ko",
        "Kyrgyz": "ky",
        "Latin": "la",
        "Latvian": "lv",
        "Lithuanian": "lt",
        "Lombard": "lmo",
        "Low Saxon": "nds",
        "Luxembourgish; Letzeburgesch": "lb",
        "Macedonian": "mk",
        "Malagasy": "mg",
        "Malay": "ms",
        "Malayalam": "ml",
        "Marathi": "mr",
        "Minangkabau": "min",
        "Nepali": "ne",
        "Newar": "new",
        "Norwegian (Bokmal)": "nob",
        "Norwegian (Nynorsk)": "nno",
        "Occitan": "oc",
        "Panjabi; Punjabi": "pa",
        "Persian": "fa",
        "Piedmontese": "pms",
        "Polish": "pl",
        "Portuguese": "pt",
        "Romanian; Moldavian; Moldovan": "ro",
        "Russian": "ru",
        "Scots": "sco",
        "Serbian": "sr",
        "Serbo-Croatian": "hbs",
        "Sicilian": "scn",
        "Slovak": "sk",
        "Slovenian": "sl",
        "South Azerbaijani": "aze",
        "Spanish": "es",
        "Sundanese": "su",
        "Swahili": "sw",
        "Swedish": "sv",
        "Tajik": "tg",
        "Tamil": "ta",
        "Tatar": "tt",
        "Telugu": "te",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Urdu": "ur",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Volapük": "vo",
        "Waray-Waray": "war",
        "Welsh": "cy",
        "Western Frisian": "fy",
        "Western Punjabi": "lah",
        "Yoruba": "yo",
    },
    "bleurt": {
        "Afrikaans": "af",
        "Albanian": "sq",
        "Amharic": "am",
        "Arabic": "ar",
        "Armenian": "hy",
        "Azerbaijani": "az",
        "Bangla": "bn",
        "Basque": "eu",
        "Belarusian": "be",
        "Bulgarian": "bg",
        "Bulgarian (romanized)": "bg-Latn",
        "Burmese": "my",
        "Catalan; Valencian": "ca",
        "Cebuano": "ceb",
        "Central Khmer": "km",
        "Chinese": "zh",
        "Chinese (romanized)": "zh-Latn",
        "Corsican": "co",
        "Czech": "cs",
        "Danish": "da",
        "Dutch; Flemish": "nl",
        "English": "en",
        "Esperanto": "eo",
        "Estonian": "et",
        "Filipino": "fil",
        "Finnish": "fi",
        "French": "fr",
        "Gaelic; Scottish Gaelic": "gd",
        "Galician": "gl",
        "Georgian": "ka",
        "German": "de",
        "Greeek": "el",
        "Greek (romanized)": "el-Latn",
        "Gujarati": "gu",
        "Haitian; Haitian Creole": "ht",
        "Hausa": "ha",
        "Hawaiian": "haw",
        "Hindi": "hi",
        "Hindi (romanized)": "hi-Latn",
        "Hmong, Mong": "hmn",
        "Hungarian": "hu",
        "Icelandic": "is",
        "Igbo": "ig",
        "Indonesian": "id",
        "Irish": "ga",
        "Italian": "it",
        "Japanese": "ja",
        "Japanese (romanized)": "ja-Latn",
        "Javanese": "jv",
        "Kannada": "kn",
        "Kazakh": "kk",
        "Korean": "ko",
        "Kurdish": "ku",
        "Kyrgyz": "ky",
        "Lao": "lo",
        "Latin": "la",
        "Latvian": "lv",
        "Lithuanian": "lt",
        "Luxembourgish; Letzeburgesch": "lb",
        "Macedonian": "mk",
        "Malagasy": "mg",
        "Malay": "ms",
        "Malayalam": "ml",
        "Maltese": "mt",
        "Maori": "mi",
        "Marathi": "mr",
        "Mongolian": "mn",
        "Nepali": "ne",
        "Norwegian": "no",
        "Nyanja": "ny",
        "Panjabi; Punjabi": "pa",
        "Persian": "fa",
        "Polish": "pl",
        "Portuguese": "pt",
        "Pushto; Pashto": "ps",
        "Romanian; Moldavian; Moldovan": "ro",
        "Russian": "ru",
        "Russian (romanized)": "ru-Latn",
        "Samoan": "sm",
        "Serbian": "sr",
        "Shona": "sn",
        "Sindhi": "sd",
        "Sinhala; Sinhalese": "si",
        "Slovak": "sk",
        "Slovenian": "sl",
        "Somali": "so",
        "Southern Sotho": "st",
        "Spanish": "es",
        "Sundanese": "su",
        "Swahili": "sw",
        "Swedish": "sv",
        "Tajik": "tg",
        "Tamil": "ta",
        "Telugu": "te",
        "Thai": "th",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Urdu": "ur",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Welsh": "cy",
        "Western Frisian": "fy",
        "Xhosa": "xh",
        "Yiddish": "yi",
        "Yoruba": "yo",
        "Zulu": "zu",
        "former Hebrew": "iw",
    },
    "comet": {
        "Afrikaans": "af",
        "Albanian": "sq",
        "Amharic": "am",
        "Arabic": "ar",
        "Armenian": "hy",
        "Assamese": "as",
        "Azerbaijani": "az",
        "Bangla": "bn",
        "Basque": "eu",
        "Belarusian": "be",
        "Bosnian": "bs",
        "Breton": "br",
        "Bulgarian": "bg",
        "Burmese": "my",
        "Catalan; Valencian": "ca",
        "Central Khmer": "km",
        "Chinese": "zh",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dutch; Flemish": "nl",
        "English": "en",
        "Esperanto": "eo",
        "Estonian": "et",
        "Finnish": "fi",
        "French": "fr",
        "Gaelic; Scottish Gaelic": "gd",
        "Galician": "gl",
        "Georgian": "ka",
        "German": "de",
        "Greeek": "el",
        "Gujarati": "gu",
        "Hausa": "ha",
        "Hebrew": "he",
        "Hindi": "hi",
        "Hungarian": "hu",
        "Icelandic": "is",
        "Indonesian": "id",
        "Irish": "ga",
        "Italian": "it",
        "Japanese": "ja",
        "Javanese": "jv",
        "Kannada": "kn",
        "Kazakh": "kk",
        "Korean": "ko",
        "Kurdish": "ku",
        "Kyrgyz": "ky",
        "Lao": "lo",
        "Latin": "la",
        "Latvian": "lv",
        "Lithuanian": "lt",
        "Macedonian": "mk",
        "Malagasy": "mg",
        "Malay": "ms",
        "Malayalam": "ml",
        "Marathi": "mr",
        "Mongolian": "mn",
        "Nepali": "ne",
        "Norwegian": "no",
        "Oriya": "or",
        "Oromo": "om",
        "Panjabi; Punjabi": "pa",
        "Persian": "fa",
        "Polish": "pl",
        "Portuguese": "pt",
        "Pushto; Pashto": "ps",
        "Romanian; Moldavian; Moldovan": "ro",
        "Russian": "ru",
        "Sanskrit": "sa",
        "Serbian": "sr",
        "Sindhi": "sd",
        "Sinhala; Sinhalese": "si",
        "Slovak": "sk",
        "Slovenian": "sl",
        "Somali": "so",
        "Spanish": "es",
        "Sundanese": "su",
        "Swahili": "sw",
        "Swedish": "sv",
        "Tagalog": "tl",
        "Tamil": "ta",
        "Telugu": "te",
        "Thai": "th",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Urdu": "ur",
        "Uyghur": "ug",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Welsh": "cy",
        "Western Frisian": "fy",
        "Xhosa": "xh",
        "Yiddish": "yi",
    },
}

SUPPORTED_LANGS_REV = {
    metric_name: {v: k for k, v in lang2key.items()} for metric_name, lang2key in SUPPORTED_LANGS.items()
}
