import importlib

import sklearn

# Supported scikit-learn minor versions. Each maps to a vendored
# ``skspark.sklearn_1_<minor>`` package holding that release's patched
# ``_search.py``. See developer-notes.md.
_SUPPORTED_MINORS = (2,)

_major, _minor = (int(part) for part in sklearn.__version__.split(".")[:2])

if _major == 1 and _minor in _SUPPORTED_MINORS:
    _search = importlib.import_module(f"skspark.sklearn_1_{_minor}._search")
    GridSearchCV = _search.GridSearchCV
    RandomizedSearchCV = _search.RandomizedSearchCV
else:
    raise NotImplementedError(
        f"scikit-learn {sklearn.__version__} is not supported; "
        f"supported versions are 1.{_SUPPORTED_MINORS[0]}-1.{_SUPPORTED_MINORS[-1]}"
    )


__all__ = [
    "GridSearchCV",
    "RandomizedSearchCV",
]
