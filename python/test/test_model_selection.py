import sklearn.model_selection
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV
from sklearn.model_selection import RandomizedSearchCV as SklearnRandomizedSearchCV

from skspark.model_selection import GridSearchCV as SparkGridSearchCV
from skspark.model_selection import RandomizedSearchCV as SparkRandomizedSearchCV
from .pyspark_test import PySparkTest


# Overwrite the sklearn GridSearch in this suite so that we can run the same
# tests with the same parameters.
class AllTests(PySparkTest):
    # After testing, make sure to revert sklearn to normal
    # (see _add_to_module())
    @classmethod
    def tearDownClass(cls):
        super(AllTests, cls).tearDownClass()
        # Restore sklearn module to the original state after done testing
        # this fixture.
        sklearn.model_selection.GridSearchCV = \
            sklearn.model_selection.GridSearchCV_original
        del sklearn.model_selection.GridSearchCV_original

        sklearn.model_selection.RandomizedSearchCV = \
            sklearn.model_selection.RandomizedSearchCV_original
        del sklearn.model_selection.RandomizedSearchCV_original


def _create_method(method):
    def do_test_expected(*kwargs):
        method()

    return do_test_expected


def _add_to_module():
    sklearn.model_selection.GridSearchCV = SparkGridSearchCV
    sklearn.model_selection.GridSearchCV_original = SklearnGridSearchCV

    sklearn.model_selection.RandomizedSearchCV = SparkRandomizedSearchCV
    sklearn.model_selection.RandomizedSearchCV_original = \
        SklearnRandomizedSearchCV

    from sklearn.model_selection.tests import test_search
    all_tests = [
        test
        for (method_name, test) in test_search.__dict__.items()
        if method_name.startswith("test_")
    ]


_add_to_module()
