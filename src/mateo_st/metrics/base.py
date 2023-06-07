from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, Type

from sacrebleu.metrics.base import Metric as SbMetric


@dataclass
class MetricOption:
    name: str
    description: str
    default: Any
    choices: Optional[Tuple] = None
    demo_choices: Optional[Tuple] = None
    types: Optional[Tuple[Type, ...]] = field(default_factory=tuple)
    empty_str_is_none: bool = False

    def __post_init__(self):
        if not self.choices and not self.types:
            raise ValueError(f"{self.name} needs at least one of 'choices' or 'types'")

        if self.choices and self.default not in self.choices:
            raise ValueError(
                f"{self.name}: the default option ('{self.default}') must be in 'choices' ('{', '.join(self.choices)}')"
            )

        if self.demo_choices is None:
            self.demo_choices = self.choices


@dataclass
class MetricMeta:
    name: str
    metric_class: Literal["baseline", "neural"]
    full_name: str
    description_html: str
    paper_url: str
    implementation_html: str
    is_default_selected: bool = False
    higher_better: bool = True
    version: Optional[str] = None
    options: Optional[Tuple[MetricOption, ...]] = field(default_factory=tuple)
    requires_source: bool = False
    sb_class: Optional[Type[SbMetric]] = None
    corpus_score_key: str = "score"
    sentences_score_key: Optional[str] = None
    segment_level: bool = True
    use_pseudo_batching: bool = True

    def postprocess_result(self, result: Dict[str, Any]):
        """Post-processes the result that is retrieved from a computed metric.

        :param result: score result (dictionary)
        :return: modified score result
        """
        return result
