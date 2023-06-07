import copy

from sklearn import datasets, svm

from skspark.model_selection import GridSearchCV, RandomizedSearchCV


def test_removing_spark_attribute():
    # train an RS
    iris = datasets.load_iris()
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = svm.SVC()
    gs = GridSearchCV(svc, parameters, spark=True)
    gs.fit(iris.data, iris.target)

    # copying twice would throw an error before
    assert hasattr(gs, "spark")
    gs_copied = copy.deepcopy(gs)
    assert not hasattr(gs_copied, "spark")
    gs_copied_again = copy.deepcopy(gs_copied)
    assert not hasattr(gs_copied_again, "spark")


def test_that_it_works_without_spark():
    # train an RS
    iris = datasets.load_iris()
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = svm.SVC()

    gs = GridSearchCV(estimator=svc, param_grid=parameters, spark=False)
    gs.fit(iris.data, iris.target)
    gs = GridSearchCV(estimator=svc, param_grid=parameters, spark=None)
    gs.fit(iris.data, iris.target)

    rs = RandomizedSearchCV(estimator=svc, param_distributions=parameters,
                            spark=False, n_iter=2)
    rs.fit(iris.data, iris.target)
    rs = RandomizedSearchCV(estimator=svc, param_distributions=parameters,
                            spark=None, n_iter=2)
    rs.fit(iris.data, iris.target)
