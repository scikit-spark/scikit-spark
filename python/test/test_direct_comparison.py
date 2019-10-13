from sklearn import datasets
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV
from sklearn.model_selection import RandomizedSearchCV \
    as SklearnRandomizedSearchCV
from sklearn.svm import SVC

from skspark.model_selection import GridSearchCV as SparkGridSearchCV
from skspark.model_selection import RandomizedSearchCV \
    as SparkRandomizedSearchCV
from test.pyspark_test import PySparkTest


class TestDirectComparison(PySparkTest):

    def test_example_grid_search(self):
        # The classic example from the sklearn documentation
        iris = datasets.load_iris()
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svr = SVC()
        clf = SklearnGridSearchCV(svr, parameters)
        clf.fit(iris.data, iris.target)

        clf2 = SparkGridSearchCV(svr, parameters, spark=True)
        clf2.fit(iris.data, iris.target)

        b1 = clf.estimator
        b2 = clf2.estimator
        self.assertEqual(b1.get_params(), b2.get_params())

    def test_example_randomized_search(self):
        # The classic example from the sklearn documentation
        iris = datasets.load_iris()
        parameters = {'kernel': ('linear', 'rbf'), 'C': range(1, 10)}
        svr = SVC()
        clf = SklearnRandomizedSearchCV(svr, parameters, random_state=4)
        clf.fit(iris.data, iris.target)

        clf2 = SparkRandomizedSearchCV(
            svr,
            parameters,
            random_state=4,
            spark=self.spark)
        clf2.fit(iris.data, iris.target)

        b1 = clf.estimator
        b2 = clf2.estimator
        self.assertEqual(b1.get_params(), b2.get_params())
