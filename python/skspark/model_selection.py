import sklearn

if sklearn.__version__.startswith("1."):
    from skspark.sklearn_1._search import GridSearchCV, RandomizedSearchCV
else:
    raise NotImplementedError("sklearn versions above 1.0 are not yet supported")


__all__ = [
    "GridSearchCV",
    "RandomizedSearchCV",
]
