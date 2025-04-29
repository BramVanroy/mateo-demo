from collections import defaultdict
from typing import Any


def build_signature(paired_bs_n: int, seed: int, library_version: str, metric_options: dict) -> str:
    """Builds a signature for a metric that does not support it out of the box (SacreBLEU does). Note that the MATEO
    version is added later on, in `03_📏_Evaluate._evaluate`

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


def merge_batched_results(results: list[dict[str, Any]]):
    """Merges the results of batched computations into a single result."""
    result = defaultdict(list)
    for batch_result in results:
        # score_key is something like "f1" or "scores"
        for score_key, scores in batch_result.items():
            if isinstance(scores, list):
                result[score_key].extend(scores)
            else:
                result[score_key].append(scores)

    return dict(result)
