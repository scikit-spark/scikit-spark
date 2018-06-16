# Spark acceleration for Scikit-Learn

This project is a major re-write of the 
[spark-sklearn](https://github.com/databricks/spark-sklearn) project, which 
seems to no longer be under development. It focuses specifically on the 
acceleration of Scikit-Learn's cross validation functionality using PySpark.

### Improvements over spark-sklearn
The functionality is based on `sklearn.model_selection` module rather than the 
deprecated and soon to be removed `sklearn.grid_search`. The new versions 
contain several nicer features and `scikit-spark` maintains full compatibility.

## Installation
The package can be installed through pip:
```bash
pip install scikit-spark
```

It has so far only been tested with Spark 2.2.0 and up, but may work with 
older versions. 

## Usage

The functionality here is meant to as closely resemble using Scikit-Learn as 
possible. The only differences are the import and that the first argument in the
constructor should be a `SparkSession` as shown below. 

This example is adapted from the Scikit-Learn documentation. It instantiates
a local `SparkSession`, and distributes the cross validation folds and 
iterations using this. In actual use, to get the benefit of this package it 
should be used distributed across several machines with Spark as running it 
locally is slower than the `Scikit-Learn` parallelisation implementation.

```python
from sklearn import svm, datasets
from pyspark.sql import SparkSession

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()

spark = SparkSession.builder\
    .master("local[*]")\
    .appName("skspark-grid-search-doctests")\
    .getOrCreate()

# How to run grid search
from skspark.model_selection import GridSearchCV

gs = GridSearchCV(spark, svc, parameters)
gs.fit(iris.data, iris.target)

# How to run random search
from skspark.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(spark, svc, parameters)
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

