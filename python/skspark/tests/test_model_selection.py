import unittest

import sklearn.model_selection
from skspark.model_selection import GridSearchCV, RandomizedSearchCV
from skspark.tests.spark_test import spark_test


# Overwrite the sklearn GridSearch in this suite so that we can run the same
# tests with the same parameters.


@spark_test
class AllTests(unittest.TestCase):

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


class GridSearchSparkWrapper(GridSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid='warn', refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise-deprecating',
                 return_train_score="warn"):
        super(GridSearchSparkWrapper, self).__init__(
            AllTests.spark, estimator, param_grid, scoring, fit_params, n_jobs,
            iid, refit, cv, verbose, pre_dispatch, error_score,
            return_train_score)


class RandomizedSearchSparkWrapper(RandomizedSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10,
                 scoring=None, fit_params=None, n_jobs=1, iid='warn',
                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score='raise-deprecating',
                 return_train_score="warn"):
        super(RandomizedSearchSparkWrapper, self).__init__(
            AllTests.spark, estimator, param_distributions, n_iter, scoring,
            fit_params, n_jobs, iid, refit, cv, verbose, pre_dispatch,
            random_state, error_score, return_train_score)


def _create_method(method):
    def do_test_expected(*kwargs):
        method()

    return do_test_expected


def _add_to_module():
    SKGridSearchCV = sklearn.model_selection.GridSearchCV
    sklearn.model_selection.GridSearchCV = GridSearchSparkWrapper
    sklearn.model_selection.GridSearchCV_original = SKGridSearchCV

    SKRandomizedSearchCV = sklearn.model_selection.RandomizedSearchCV
    sklearn.model_selection.RandomizedSearchCV = RandomizedSearchSparkWrapper
    sklearn.model_selection.RandomizedSearchCV_original = SKRandomizedSearchCV

    from sklearn.model_selection.tests import test_search
    all_methods = [(mname, method) for (mname, method) in
                   test_search.__dict__.items()
                   if mname.startswith("test_")]

    for name, method in all_methods:
        method_for_test = _create_method(method)
        method_for_test.__name__ = name
        setattr(AllTests, method.__name__, method_for_test)


_add_to_module()
