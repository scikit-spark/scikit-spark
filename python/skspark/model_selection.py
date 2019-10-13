import sklearn


if sklearn.__version__.startswith("0.19."):
    from skspark.sklearn_0_19.model_selection import GridSearchCV, RandomizedSearchCV
elif sklearn.__version__.startswith("0.20."):
    from skspark.sklearn_0_20._search import GridSearchCV, RandomizedSearchCV
