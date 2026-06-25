"""Corporate credit-rating prediction — core package.

Compares three tabular approaches on a corporate credit-rating task:
XGBoost (manual baseline), TabPFN-3 (zero-shot foundation model), and
AutoGluon (AutoML).

Typical usage from a notebook::

    from credit_rating import config, data, models, experiments, evaluate, plots
"""
from . import config, data, evaluate, experiments, models, plots  # noqa: F401

__all__ = ["config", "data", "models", "evaluate", "experiments", "plots"]
__version__ = "1.0.0"
