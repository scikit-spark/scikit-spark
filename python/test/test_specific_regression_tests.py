import sys
from types import GeneratorType

import numpy as np
from numpy.testing import assert_array_almost_equal
import sklearn
from sklearn.datasets import make_classification
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import KFold
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import MockClassifier, \
    FailingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.mocking import CheckingClassifier
from sklearn.utils.testing import assert_true, assert_raise_message, assert_raises, assert_warns

from skspark.model_selection import GridSearchCV, RandomizedSearchCV

if sys.version_info[0] > 2:
    from . pyspark_test import PySparkTest
else:
    from pyspark_test import PySparkTest


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


def check_hyperparameter_searcher_with_fit_params(klass, **klass_kwargs):
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    clf = CheckingClassifier(expected_fit_params=['spam', 'eggs'])
    searcher = klass(clf, {'foo_param': [1, 2, 3]}, cv=2, **klass_kwargs)

    # The CheckingClassifier generates an assertion error if
    # a parameter is missing or has length != len(X).
    assert_raise_message(AssertionError,
                         "Expected fit parameter(s) ['eggs'] not seen.",
                         searcher.fit, X, y, spam=np.ones(10))
    assert_raise_message(AssertionError,
                         "Fit parameter spam has length 1; expected 4.",
                         searcher.fit, X, y, spam=np.ones(1),
                         eggs=np.zeros(10))
    searcher.fit(X, y, spam=np.ones(10), eggs=np.zeros(10))


class TestSpecificRegressionTests(PySparkTest):
    """This class contains some select tests taken from scikit-learn."""
    def test_trivial_cv_results_attr(self):
        # Test search over a "grid" with only one point.
        # Non-regression test: grid_scores_ wouldn't be set by GridSearchCV.
        clf = MockClassifier()
        grid_search = GridSearchCV(clf, {'foo_param': [1]})
        grid_search.fit(X, y)
        assert_true(hasattr(grid_search, "cv_results_"))

        # random_search = RandomizedSearchCV(clf, {'foo_param': [0]}, n_iter=1)
        # random_search.fit(X, y)
        # assert_true(hasattr(grid_search, "cv_results_"))

    def test_random_search_with_fit_params(self):
        check_hyperparameter_searcher_with_fit_params(
            RandomizedSearchCV, n_iter=1)

    def test_grid_search_precomputed_kernel_error_nonsquare(self):
        # Test that grid search returns an error with a non-square precomputed
        # training kernel matrix
        K_train = np.zeros((10, 20))
        y_train = np.ones((10,))
        clf = SVC(kernel='precomputed')
        cv = GridSearchCV(clf, {'C': [0.1, 1.0]})
        assert_raises(ValueError, cv.fit, K_train, y_train)

    def test_grid_search_cv_splits_consistency(self):
        # Check if a one time iterable is accepted as a cv parameter.
        n_samples = 100
        n_splits = 5
        X, y = make_classification(n_samples=n_samples, random_state=0)

        gs = GridSearchCV(LinearSVC(random_state=0),
                          param_grid={'C': [0.1, 0.2, 0.3]},
                          cv=OneTimeSplitter(n_splits=n_splits,
                                             n_samples=n_samples))
        gs.fit(X, y)

        gs2 = GridSearchCV(LinearSVC(random_state=0),
                           param_grid={'C': [0.1, 0.2, 0.3]},
                           cv=KFold(n_splits=n_splits))
        gs2.fit(X, y)

        # Give generator as a cv parameter
        assert_true(isinstance(KFold(n_splits=n_splits,
                                     shuffle=True, random_state=0).split(X, y),
                               GeneratorType))
        gs3 = GridSearchCV(LinearSVC(random_state=0),
                           param_grid={'C': [0.1, 0.2, 0.3]},
                           cv=KFold(n_splits=n_splits, shuffle=True,
                                    random_state=0).split(X, y))
        gs3.fit(X, y)

        gs4 = GridSearchCV(LinearSVC(random_state=0),
                           param_grid={'C': [0.1, 0.2, 0.3]},
                           cv=KFold(n_splits=n_splits, shuffle=True,
                                    random_state=0))
        gs4.fit(X, y)

        def _pop_time_keys(cv_results):
            for key in ('mean_fit_time', 'std_fit_time',
                        'mean_score_time', 'std_score_time'):
                cv_results.pop(key)
            return cv_results

        # Check if generators are supported as cv and
        # that the splits are consistent
        np.testing.assert_equal(_pop_time_keys(gs3.cv_results_),
                                _pop_time_keys(gs4.cv_results_))

        # OneTimeSplitter is a non-re-entrant cv where split can be called only
        # once if ``cv.split`` is called once per param setting in GridSearchCV.fit
        # the 2nd and 3rd parameter will not be evaluated as no train/test indices
        # will be generated for the 2nd and subsequent cv.split calls.
        # This is a check to make sure cv.split is not called once per param
        # setting.
        np.testing.assert_equal({k: v for k, v in gs.cv_results_.items()
                                 if not k.endswith('_time')},
                                {k: v for k, v in gs2.cv_results_.items()
                                 if not k.endswith('_time')})

        # Check consistency of folds across the parameters
        gs = GridSearchCV(LinearSVC(random_state=0),
                          param_grid={'C': [0.1, 0.1, 0.2, 0.2]},
                          cv=KFold(n_splits=n_splits, shuffle=True))
        gs.fit(X, y)

        # As the first two param settings (C=0.1) and the next two param
        # settings (C=0.2) are same, the test and train scores must also be
        # same as long as the same train/test indices are generated for all
        # the cv splits, for both param setting
        for score_type in ('train', 'test'):
            per_param_scores = {}
            for param_i in range(4):
                per_param_scores[param_i] = list(
                    gs.cv_results_['split%d_%s_score' % (s, score_type)][
                        param_i]
                    for s in range(5))

            assert_array_almost_equal(per_param_scores[0],
                                      per_param_scores[1])
            assert_array_almost_equal(per_param_scores[2],
                                      per_param_scores[3])

    def test_grid_search_failing_classifier(self):
        """The grid_scores_ attribute was removed in 0.20, so the test is only relevant for 0.19"""
        if sklearn.__version__.startswith("0.19."):
            # GridSearchCV with on_error != 'raise'
            # Ensures that a warning is raised and score reset where appropriate.

            X, y = make_classification(n_samples=20, n_features=10, random_state=0)

            clf = FailingClassifier()

            # refit=False because we only want to check that errors caused by fits
            # to individual folds will be caught and warnings raised instead. If
            # refit was done, then an exception would be raised on refit and not
            # caught by grid_search (expected behavior), and this would cause an
            # error in this test.
            gs = GridSearchCV(clf, [{'parameter': [0, 1, 2]}],
                              scoring='accuracy', refit=False, error_score=0.0)

            assert_warns(FitFailedWarning, gs.fit, X, y)

            # Ensure that grid scores were set to zero as required for those fits
            # that are expected to fail.
            assert all(np.all(this_point.cv_validation_scores == 0.0)
                       for this_point in gs.grid_scores_
                       if this_point.parameters['parameter'] ==
                       FailingClassifier.FAILING_PARAMETER)

            gs = GridSearchCV(clf, [{'parameter': [0, 1, 2]}],
                              scoring='accuracy', refit=False,
                              error_score=float('nan'))
            assert_warns(FitFailedWarning, gs.fit, X, y)
            assert all(np.all(np.isnan(this_point.cv_validation_scores))
                       for this_point in gs.grid_scores_
                       if this_point.parameters['parameter'] ==
                       FailingClassifier.FAILING_PARAMETER)
