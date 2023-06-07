# Spark acceleration for Scikit-Learn

This project is a major re-write of the 
[spark-sklearn](https://github.com/databricks/spark-sklearn) project, which 
seems to no longer be under development. It focuses specifically on the 
acceleration of Scikit-Learn's cross validation functionality using PySpark.

### Improvements over spark-sklearn
`scikit-spark` supports `scikit-learn` versions past 0.19, `spark-sklearn` [have stated that they are probably not 
going to support newer versions](https://github.com/databricks/spark-sklearn/issues/113).

The functionality in `scikit-spark` is based on `sklearn.model_selection` module rather than the 
deprecated and soon to be removed `sklearn.grid_search`. The new `model_selection` versions 
contain several nicer features and `scikit-spark` maintains full compatibility.

## Installation
The package can be installed through pip:
```bash
pip install scikit-spark

# or to also install with pyspark
pip install scikit-spark[spark]
```

It is currently tested with Spark 3.3.2 and 3.4.0, but should work with
older versions (e.g. Spark 2.4.x). Python versions 3.8 and up are also
currently tested, though 3.11 is only supported with PySpark 3.4.0.

### Supported scikit-learn versions
- 1.2.2

### Older versions of scikit-learn

The following pre-1.0 releases of scikit-learn are supported by `scikit-spark==0.4.0`:

- 0.19
- 0.20
- 0.21 (Python 3 only)
- 0.22 (Python 3 only)

## Usage

The functionality here is meant to as closely resemble using Scikit-Learn as 
possible. By default (with `spark=True`) the `SparkSession` is obtained
internally by calling `SparkSession.builder.getOrCreate()`, so the instantiation
and calling of the functions is the same (You will preferably have already 
created a `SparkSession`). 

This example is adapted from the Scikit-Learn documentation. It instantiates
a local `SparkSession`, and distributes the cross validation folds and 
iterations using this. In actual use, to get the benefit of this package it 
should be used distributed across several machines with Spark as running it 
locally is slower than the `Scikit-Learn` parallelisation implementation.

```python
from sklearn import svm, datasets
from pyspark.sql import SparkSession

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1, 10, 100]}
svc = svm.SVC()

spark = SparkSession.builder\
    .master("local[*]")\
    .appName("skspark-grid-search-doctests")\
    .getOrCreate()

# How to run grid search
from skspark.model_selection import GridSearchCV

gs = GridSearchCV(svc, parameters, spark=spark)
gs.fit(iris.data, iris.target)

# How to run random search
from skspark.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(svc, parameters, spark=spark)
rs.fit(iris.data, iris.target)
```

## Current and upcoming functionality
- Current
    - model_selection.RandomizedSearchCV
    - model_selection.GridSearchCV
- Upcoming
    - model_selection.cross_val_predict
    - model_selection.cross_val_score

*The docstrings are modifications of the Scikit-Learn ones and are still being
converted to specifically refer to this project.* 

## Performance optimisations 

### Reducing RAM usage 
*Coming soon*

