"""
Python 3 throws ResourceWarning s for unclosed sockets.
There is one test inside sklearn which is supposed to not return any warnings,
but does throw an unrelated resource warning.

This file has this test whilst swallowing the resource warning.
"""

import sys
import warnings
from unittest import skipIf

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import clean_warning_registry

from skspark.model_selection import GridSearchCV, RandomizedSearchCV
from .sklearn_version_specific_utils import sklearn_is_0_21, sklearn_version_is

if sys.version_info[0] > 2:
    from . pyspark_test import PySparkTest
else:
    from pyspark_test import PySparkTest


def assert_no_warnings(func, *args, **kw):
    # very important to avoid uncontrolled state propagation
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        result = func(*args, **kw)
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w
                 if e.category is not np.VisibleDeprecationWarning]

        # skip ResourceWarning socket warnings (Python 3 only)
        if sys.version_info[0] > 2:
            w = [e for e in w if e.category != ResourceWarning]

        if len(w) > 0:
            raise AssertionError("Got warnings when calling %s: [%s]"
                                 % (func.__name__,
                                    ', '.join(str(warning) for warning in w)))
    return result


class ResourceWarningTests(PySparkTest):
    """This class contains some tests which break due to returning
    ResourceWarnings. The sklearn versions are skipped in favour of these"""

    @skipIf(not sklearn_version_is("0.19"), "0.19 implementation of test")
    def test_return_train_score_warn_0_19(self):
        # Test that warnings are raised. Will be removed in 0.21

        X = np.arange(100).reshape(10, 10)
        y = np.array([0] * 5 + [1] * 5)
        grid = {'C': [1, 2]}

        estimators = [GridSearchCV(LinearSVC(random_state=0), grid),
                      RandomizedSearchCV(LinearSVC(random_state=0), grid,
                                         n_iter=2)]

        result = {}
        for estimator in estimators:
            for val in [True, False, 'warn']:
                estimator.set_params(return_train_score=val)
                result[val] = assert_no_warnings(estimator.fit, X, y).cv_results_

        train_keys = ['split0_train_score', 'split1_train_score',
                      'split2_train_score', 'mean_train_score', 'std_train_score']
        for key in train_keys:
            msg = (
                'You are accessing a training score ({!r}), '
                'which will not be available by default '
                'any more in 0.21. If you need training scores, '
                'please set return_train_score=True').format(key)
            train_score = assert_warns_message(FutureWarning, msg,
                                               result['warn'].get, key)
            assert np.allclose(train_score, result[True][key])
            assert key not in result[False]

        for key in result['warn']:
            if key not in train_keys:
                assert_no_warnings(result['warn'].get, key)

    @skipIf(not sklearn_version_is("0.20"), "0.20 version of test")
    def test_return_train_score_warn(self):
        from sklearn.utils.testing import ignore_warnings
        # Test that warnings are raised. Will be removed in 0.21

        X = np.arange(100).reshape(10, 10)
        y = np.array([0] * 5 + [1] * 5)
        grid = {'C': [1, 2]}

        estimators = [GridSearchCV(LinearSVC(random_state=0), grid,
                                   iid=False, cv=3),
                      RandomizedSearchCV(LinearSVC(random_state=0), grid,
                                         n_iter=2, iid=False, cv=3)]

        result = {}
        for estimator in estimators:
            for val in [True, False, 'warn']:
                estimator.set_params(return_train_score=val)
                fit_func = ignore_warnings(estimator.fit,
                                           category=ConvergenceWarning)
                result[val] = assert_no_warnings(fit_func, X, y).cv_results_

        train_keys = ['split0_train_score', 'split1_train_score',
                      'split2_train_score', 'mean_train_score', 'std_train_score']
        for key in train_keys:
            msg = (
                'You are accessing a training score ({!r}), '
                'which will not be available by default '
                'any more in 0.21. If you need training scores, '
                'please set return_train_score=True').format(key)
            train_score = assert_warns_message(FutureWarning, msg,
                                               result['warn'].get, key)
            assert np.allclose(train_score, result[True][key])
            assert key not in result[False]

        for key in result['warn']:
            if key not in train_keys:
                assert_no_warnings(result['warn'].get, key)

    @skipIf(not (sklearn_version_is("0.20") or sklearn_is_0_21()), "test only applicable to sklearn 0.20 and 0.21")
    def test_deprecated_grid_search_iid(self):
        depr_message = ("The default of the `iid` parameter will change from True "
                        "to False in version 0.22")
        X, y = make_blobs(n_samples=54, random_state=0, centers=2)
        grid = GridSearchCV(SVC(gamma='scale', random_state=0),
                            param_grid={'C': [10]}, cv=3)
        # no warning with equally sized test sets
        assert_no_warnings(grid.fit, X, y)

        grid = GridSearchCV(SVC(gamma='scale', random_state=0),
                            param_grid={'C': [10]}, cv=5)
        # warning because 54 % 5 != 0
        assert_warns_message(DeprecationWarning, depr_message, grid.fit, X, y)

        grid = GridSearchCV(SVC(gamma='scale', random_state=0),
                            param_grid={'C': [10]}, cv=2)
        # warning because stratification into two classes and 27 % 2 != 0
        assert_warns_message(DeprecationWarning, depr_message, grid.fit, X, y)

        grid = GridSearchCV(SVC(gamma='scale', random_state=0),
                            param_grid={'C': [10]}, cv=KFold(2))
        # no warning because no stratification and 54 % 2 == 0
        assert_no_warnings(grid.fit, X, y)
