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


def pytest_collection_modifyitems(session, config, items):
    """Remove tests from the collection

    The following tests (from sklearn) are removed:
    * test_search_cv_verbose_3 - this checks the output from stderr which is swallowed by the pyspark executors

    Some skips are scikit-learn-version specific (older releases behave
    differently against the modern numpy/scipy stack we test on); these are
    gated on the installed minor version.

    See https://docs.pytest.org/en/6.2.x/reference.html#pytest.hookspec.pytest_collection_modifyitems
    """
    import sklearn

    minor = int(sklearn.__version__.split(".")[1])

    tests_to_skip = [
        "test_search_cv_verbose_3",
        # Stateful scorer counts its own calls to emit a NaN on every 5th call;
        # distributing the CV fits across Spark workers pickles the scorer per
        # worker, resetting the counter, so the warning count differs.
        "test_searchcv_raise_warning_with_non_finite_score[RandomizedSearchCV",
    ]

    if minor == 1:
        # sklearn 1.1's _format_results ranks failed (NaN) fits with
        # `rankdata(-means).astype(int32)`; modern scipy's rankdata propagates
        # NaN, so the NaN cast to int32 becomes INT32_MIN instead of the worst
        # rank. This is an upstream 1.1 + modern-scipy bug (fixed by the
        # NaN-aware ranking added in 1.2), unrelated to Spark.
        tests_to_skip.append("test_grid_search_failing_classifier")
        # The GridSearchCV variant of the stateful-scorer test above also trips
        # the per-worker counter reset on 1.1 (1.2+ happen to pass it).
        tests_to_skip.append(
            "test_searchcv_raise_warning_with_non_finite_score[GridSearchCV"
        )

    skip_listed = pytest.mark.skip(reason="skipped")
    for item in items:
        if item.name.startswith(tuple(tests_to_skip)):
            item.add_marker(skip_listed)
