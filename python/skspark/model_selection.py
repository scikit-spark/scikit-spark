import sklearn

if sklearn.__version__.startswith("1."):
    from skspark.sklearn_1._search import GridSearchCV, RandomizedSearchCV
else:
    raise NotImplementedError("Only sklearn 1.x is supported")


__all__ = [
    "GridSearchCV",
    "RandomizedSearchCV",
]
