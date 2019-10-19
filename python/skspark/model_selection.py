import sklearn
import sys


if sklearn.__version__.startswith("0.19."):
    from skspark.sklearn_0_19.model_selection import GridSearchCV, RandomizedSearchCV
elif sklearn.__version__.startswith("0.20."):
    from skspark.sklearn_0_20._search import GridSearchCV, RandomizedSearchCV
elif sklearn.__version__.startswith("0.21."):
    if sys.version_info[0] < 3:
        raise NotImplementedError(
            "sklearn 0.21 does not support Python 2, please downgrade sklearn or upgrade Python")
    from skspark.sklearn_0_21._search import GridSearchCV, RandomizedSearchCV
