from unittest import skipIf

import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.svm import LinearSVC

from test.pyspark_test import PySparkTest
from test.sklearn_version_specific_utils import sklearn_is_0_21


class TestSpecificRegressionTests(PySparkTest):
    @skipIf(not sklearn_is_0_21(), "test only for sklearn 0.21")
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

    @skipIf(not sklearn_is_0_21(), "test only for sklearn 0.21")
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
