from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple, Type

from sacrebleu.metrics.base import Metric as SbMetric


@dataclass
class MetricOption:
    name: str
    description: str
    default: Any
    choices: Optional[Tuple] = field(default_factory=tuple)
    types: Optional[Tuple[Type, ...]] = field(default_factory=tuple)
    empty_str_is_none: bool = False

    def __post_init__(self):
        if not self.choices and not self.types:
            raise ValueError(f"{self.name} needs at least one of 'choices' or 'types'")

        if self.choices and self.default not in self.choices:
            raise ValueError(
                f"{self.name}: the default option ('{self.default}') must be in 'choices' ('{', '.join(self.choices)}')"
            )


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
