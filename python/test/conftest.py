import logging
from functools import wraps
from unittest.mock import patch

import pytest
from pyspark.sql import SparkSession

from skspark.model_selection import RandomizedSearchCV as SkSparkRandomizedSearchCV
from skspark.model_selection import GridSearchCV as SkSparkGridSearchCV


@pytest.fixture(scope="session")
def spark():
    spark_session = SparkSession.builder\
        .master("local[*]")\
        .appName("scikit-spark-tests")\
        .getOrCreate()

    logger = logging.getLogger("py4j")
    logger.setLevel(logging.WARN)

    return spark_session


def monkey_patch_decorator(func):
    """Decorator that patches sklearn imports with those from this module"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch.multiple(
            "sklearn.model_selection.tests.test_search",
            RandomizedSearchCV=SkSparkRandomizedSearchCV,
            GridSearchCV=SkSparkGridSearchCV,
        ):
            func(*args, **kwargs)

    return wrapper


def pytest_runtest_call(item):
    """Patch the call that actually runs the tests to apply the monkey patch decorator

    This ensures that when we import the sklearn tests there are all automatically
    monkey patched to use this packages' RandomSearchCV and GridSearchCV classes.
    This means we can run all of sklearn's tests directly against the class instances
    here without having to rewrite them.

    See https://docs.pytest.org/en/6.2.x/reference.html#pytest.hookspec.pytest_runtest_call
    """
    # Only automatically decorate the tests that are hijacked from sklearn iteself
    if item.parent.name == "test_sklearn.py":
        testfunction = item.obj
        item.obj = monkey_patch_decorator(testfunction)
