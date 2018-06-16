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
*Coming soon*

## Usage


### Grid search
```python
from sklearn import svm, datasets
from skspark.model_selection import GridSearchCV
from pyspark.sql import SparkSession

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()

spark = SparkSession.builder\
    .master("local[*]")\
    .appName("skspark-grid-search-doctests")\
    .getOrCreate()

clf = GridSearchCV(spark, svc, parameters)
clf.fit(iris.data, iris.target)
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

