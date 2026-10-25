"""Korean news headline category classifier.

Public entry points live in submodules so that importing the package stays
cheap (no torch/transformers import until a model is actually built):

    from news_classifier.config import LABELS
    from news_classifier.predict import NewsClassifier
"""

from __future__ import annotations

from .config import (
    DEFAULT_QUESTION,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    ModelConfig,
    TrainConfig,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_QUESTION",
    "ID2LABEL",
    "LABEL2ID",
    "LABELS",
    "ModelConfig",
    "TrainConfig",
    "__version__",
]
