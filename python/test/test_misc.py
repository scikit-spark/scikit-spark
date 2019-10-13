import copy
import sys

from sklearn import svm, datasets
from sklearn.utils.testing import assert_true, assert_false

from skspark.model_selection import GridSearchCV

if sys.version_info[0] > 2:
    from . pyspark_test import PySparkTest
else:
    from pyspark_test import PySparkTest


class MiscTests(PySparkTest):

    def test_removing_spark_attribute(self):
        # train an RS
        iris = datasets.load_iris()
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svc = svm.SVC()
        gs = GridSearchCV(svc, parameters, spark=True)
        gs.fit(iris.data, iris.target)

        # copying twice would throw an error before
        assert_true(hasattr(gs, "spark"))
        gs_copied = copy.deepcopy(gs)
        assert_false(hasattr(gs_copied, "spark"))
        gs_copied_again = copy.deepcopy(gs_copied)
        assert_false(hasattr(gs_copied_again, "spark"))

    def test_that_it_works_without_spark(self):
        # train an RS
        iris = datasets.load_iris()
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svc = svm.SVC()
        gs = GridSearchCV(estimator=svc, param_grid=parameters, spark=False)
        gs.fit(iris.data, iris.target)
        gs = GridSearchCV(estimator=svc, param_grid=parameters, spark=None)
        gs.fit(iris.data, iris.target)
        # TODO - add test conditions
