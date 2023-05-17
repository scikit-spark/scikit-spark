import sklearn


def sklearn_version_is(version):
    if sklearn.__version__.startswith(version):
        return True
    return False


def sklearn_is_at_least(version):
    if sklearn.__version__ >= version:
        return True
    return False


def get_refactored_tests_to_skip():
    """These tests have been edited in order to work with spark.
    They have been moved into this repo e.g. in resource_warning_tests.py"""
    if sklearn_version_is("0.19"):
        return [
            "test_return_train_score_warn",  # moved to resource_warning_tests.py
        ]
    elif sklearn_version_is("0.20"):
        return [
            "test_return_train_score_warn",  # moved to resource_warning_tests.py
            "test_deprecated_grid_search_iid",  # moved to resource_warning_tests.py
            "test_validate_parameter_grid_input"  # a function, not a test
        ]
    elif sklearn_version_is("0.21"):
        return [
            "test_refit_callable_out_bound",  # parameterized test, moved to test_parameterised_tests
            "test_deprecated_grid_search_iid",  # moved to resource_warning_tests.py
            "test_validate_parameter_grid_input",  # parameterized test, moved to test_parameterised_tests
        ]
    elif sklearn_version_is("0.22"):
        return [
            "test_refit_callable_out_bound",  # parameterized test, moved to test_parameterised_tests
            "test_deprecated_grid_search_iid",  # moved to resource_warning_tests.py
            "test_validate_parameter_grid_input",  # parameterized test, moved to test_parameterised_tests
            "test_SearchCV_with_fit_params",  # moved to test_parameterised_tests
            "test_scalar_fit_param",  # moved to test_parameterised_tests
            "test_scalar_fit_param_compat",  # moved to test_parameterised_tests
            "test_search_default_iid",  # moved to test_parameterised_tests
            "test_validate_parameter_input",  # moved to test_parameterised_tests
        ]
    elif sklearn_version_is("1."):
        # TODO
        return []
    else:
        raise NotImplementedError(
            "Unsupported sklearn version {}".format(sklearn.__version__))
