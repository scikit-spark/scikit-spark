from functools import partial
from test.pyspark_test import PySparkTest
from test.sklearn_version_specific_utils import sklearn_is_at_least, sklearn_version_is
from unittest import skipIf

import pytest
from scipy.stats import uniform
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler, train_test_split
from sklearn.svm import LinearSVC
from skspark.model_selection import GridSearchCV, RandomizedSearchCV


class _FitParamClassifier(SGDClassifier):

    def fit(self, X, y, sample_weight=None, tuple_of_arrays=None,
            scalar_param=None, callable_param=None):
        super().fit(X, y, sample_weight=sample_weight)
        assert scalar_param > 0
        assert callable(callable_param)

        # The tuple of arrays should be preserved as tuple.
        assert isinstance(tuple_of_arrays, tuple)
        assert tuple_of_arrays[0].ndim == 2
        assert tuple_of_arrays[1].ndim == 1
        return self


def _fit_param_callable():
    pass


class TestParameterisedTests(PySparkTest):
    @skipIf(not sklearn_is_at_least("0.21"), "test for sklearn 0.21 and above")
    def test_refit_callable_out_bound(self):
        """
        Test implementation catches the errors when 'best_index_' returns an
        out of bound result.
        """
        out_bound_values = [-1, 2]
        search_cvs = [RandomizedSearchCV, GridSearchCV]

        for out_bound_value, search_cv in zip(out_bound_values, search_cvs):
            def refit_callable_out_bound(cv_results):
                """
                A dummy function tests when returned 'best_index_' is out of bounds.
                """
                return out_bound_value

            X, y = make_classification(n_samples=100, n_features=4,
                                       random_state=42)

            clf = search_cv(LinearSVC(random_state=42), {'C': [0.1, 1]},
                            scoring='precision', refit=refit_callable_out_bound, cv=5)
            with pytest.raises(IndexError, match='best_index_ index out of range'):
                clf.fit(X, y)

    @skipIf(not sklearn_is_at_least("0.21") or sklearn_version_is("1."), "test for sklearn >=0.21,<1.0")
    def test_validate_parameter_grid_input_wrapper(self):
        def test_validate_parameter_grid_input(input, error_type, error_message):
            with pytest.raises(error_type, match=error_message):
                ParameterGrid(input)

        parameters = [
            (0, TypeError, r'Parameter grid is not a dict or a list \(0\)'),
            ([{'foo': [0]}, 0], TypeError, r'Parameter grid is not a dict \(0\)'),
            ({'foo': 0}, TypeError, "Parameter grid value is not iterable "
                                    r"\(key='foo', value=0\)")
        ]
        for input, error_type, error_message in parameters:
            test_validate_parameter_grid_input(input, error_type, error_message)

    @skipIf(not sklearn_is_at_least("0.22") or sklearn_version_is("1."), "test for sklearn >=0.21,<1.0")
    def test_validate_parameter_input_wrapper(self):
        from sklearn.model_selection.tests.test_search import test_validate_parameter_input

        parameters = [
            (0, TypeError, r'Parameter .* is not a dict or a list \(0\)'),
            ([{'foo': [0]}, 0], TypeError, r'Parameter .* is not a dict \(0\)'),
            ({'foo': 0}, TypeError, "Parameter.* value is not iterable .*" r"\(key='foo', value=0\)")
        ]

        for klass in [ParameterGrid, partial(ParameterSampler, n_iter=10)]:
            for input, error_type, error_message in parameters:
                test_validate_parameter_input(klass, input, error_type, error_message)

    @skipIf(not sklearn_is_at_least("0.22"), "test for sklearn 0.22 and above")
    def test_search_default_iid_wrapper(self):
        from sklearn.model_selection.tests.test_search import test_search_default_iid

        parameters = [
            (GridSearchCV, {'param_grid': {'C': [1, 10]}}),
            (RandomizedSearchCV, {'param_distributions': {'C': [1, 10]}, 'n_iter': 2})
        ]

        for SearchCV, specialized_params in parameters:
            test_search_default_iid(SearchCV, specialized_params)

    @skipIf(not sklearn_is_at_least("0.22"), "test for sklearn 0.22 and above")
    def test_SearchCV_with_fit_params_wrapper(self):
        from sklearn.model_selection.tests.test_search import test_SearchCV_with_fit_params

        for SearchCV in [GridSearchCV, RandomizedSearchCV]:
            test_SearchCV_with_fit_params(SearchCV)

    @skipIf(not sklearn_is_at_least("0.22"), "test for sklearn 0.22 and above")
    def test_scalar_fit_param_wrapper(self):
        from sklearn.model_selection.tests.test_search import test_scalar_fit_param

        for SearchCV, param_search in [
            (GridSearchCV, {'a': [0.1, 0.01]}),
            (RandomizedSearchCV, {'a': uniform(1, 3)})
        ]:
            test_scalar_fit_param(SearchCV, param_search)

    @skipIf(not sklearn_is_at_least("0.22"), "test for sklearn 0.22 and above")
    def test_scalar_fit_param_compat_wrapper(self):

        def test_scalar_fit_param_compat(SearchCV, param_search):
            """
            The other test can't be pickled as the _FitParamClassifier is not accessible globally.
            So the code has to be duplicated here rather than imported
            """
            X_train, X_valid, y_train, y_valid = train_test_split(
                *make_classification(random_state=42), random_state=42
            )

            model = SearchCV(
                _FitParamClassifier(), param_search
            )

            fit_params = {
                'tuple_of_arrays': (X_valid, y_valid),
                'callable_param': _fit_param_callable,
                'scalar_param': 42,
            }
            model.fit(X_train, y_train, **fit_params)

        for SearchCV, param_search in [
            (GridSearchCV, {'alpha': [0.1, 0.01]}),
            (RandomizedSearchCV, {'alpha': uniform(0.01, 0.1)})
        ]:
            test_scalar_fit_param_compat(SearchCV, param_search)
