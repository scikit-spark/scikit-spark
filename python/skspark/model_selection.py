import sklearn


if sklearn.__version__.startswith("0.19."):
    from skspark.sklearn_0_19.model_selection import GridSearchCV, RandomizedSearchCV
