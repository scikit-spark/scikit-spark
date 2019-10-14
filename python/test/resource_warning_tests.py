# """
# Python 3 throws ResourceWarning s for unclosed sockets.
# There is one test inside sklearn which is supposed to not return any warnings,
# but does throw an unrelated resource warning.
#
# This file has this test whilst swallowing the resource warning.
# """
# import sys
# import warnings
#
# import numpy as np
# from sklearn.svm import LinearSVC
# from sklearn.utils.testing import assert_warns_message, clean_warning_registry
#
# from skspark.model_selection import GridSearchCV, RandomizedSearchCV
#
# if sys.version_info[0] > 2:
#     from . pyspark_test import PySparkTest
# else:
#     from pyspark_test import PySparkTest
#
#
# def assert_no_warnings(func, *args, **kw):
#     # very important to avoid uncontrolled state propagation
#     clean_warning_registry()
#     with warnings.catch_warnings(record=True) as w:
#         warnings.simplefilter('always')
#
#         result = func(*args, **kw)
#         if hasattr(np, 'VisibleDeprecationWarning'):
#             # Filter out numpy-specific warnings in numpy >= 1.9
#             w = [e for e in w
#                  if e.category is not np.VisibleDeprecationWarning]
#
#         # skip ResourceWarning socket warnings
#         w = [e for e in w if e.category is not ResourceWarning]
#
#         if len(w) > 0:
#             raise AssertionError("Got warnings when calling %s: [%s]"
#                                  % (func.__name__,
#                                     ', '.join(str(warning) for warning in w)))
#     return result
#
#
# class ResourceWarningTests(PySparkTest):
#     """This class contains some tests which break due to returning
#     ResourceWarnings."""
#     def test_return_train_score_warn(self):
#         # Test that warnings are raised. Will be removed in 0.21
#
#         X = np.arange(100).reshape(10, 10)
#         y = np.array([0] * 5 + [1] * 5)
#         grid = {'C': [1, 2]}
#
#         estimators = [
#             GridSearchCV(LinearSVC(random_state=0), grid, cv=3, spark=True),
#             RandomizedSearchCV(
#                 LinearSVC(random_state=0), grid, n_iter=2, cv=3, spark=True)
#         ]
#
#         result = {}
#         for estimator in estimators:
#             for val in [True, False, 'warn']:
#                 estimator.set_params(return_train_score=val)
#                 result[val] = assert_no_warnings(estimator.fit, X,
#                                                  y).cv_results_
#
#         train_keys = ['split0_train_score', 'split1_train_score',
#                       'split2_train_score', 'mean_train_score',
#                       'std_train_score']
#         for key in train_keys:
#             msg = (
#                 'You are accessing a training score ({!r}), '
#                 'which will not be available by default '
#                 'any more in 0.21. If you need training scores, '
#                 'please set return_train_score=True').format(key)
#             train_score = assert_warns_message(FutureWarning, msg,
#                                                result['warn'].get, key)
#             assert np.allclose(train_score, result[True][key])
#             assert key not in result[False]
#
#         for key in result['warn']:
#             if key not in train_keys:
#                 assert_no_warnings(result['warn'].get, key)
