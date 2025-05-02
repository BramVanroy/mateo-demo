import json
from os import PathLike
from pathlib import Path
from mateo_st.metrics.metrics_constants import NEURAL_METRIC_GETTERS
from mateo_st.translator import Translator

def main(config_file: str | PathLike, precache_translation: bool = False) -> None:
    metric_options = json.loads(Path(config_file).read_text(encoding="utf-8"))
    for metric_name, metric_options_list in metric_options.items():
        if metric_name in NEURAL_METRIC_GETTERS:
            for metric_options in metric_options_list:
                metric_getter = NEURAL_METRIC_GETTERS[metric_name]
                metric_getter(**metric_options)
        else:
            print(f"Metric {metric_name} is not a neural metric or not supported. Ignoring...")
    
    if precache_translation:
        # Downloads the default translation model for the app
        Translator(
            src_lang="Dutch",
            tgt_lang="English",
        )


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Precache neural metrics so that they are immediately available in the app and do not require an additional download.")
    cparser.add_argument(
        "--config_file",
        required=True,
        type=str,
        help="Path to the config file. Expected to be in JSON format where the main key is the metric name and the value is a list of dictionaries (one per model). See configs/precache-demo.json for an example. For all possible values for each metric, see the code in src/mateo_st/metrics",
    )
    cparser.add_argument(
        "--precache_translation",
        action="store_true",
        help="Also precache the translation model.",
    )
    cargs = cparser.parse_args()
    main(cargs.config_file, cargs.precache_translation)