# Developer notes

## How to add support for a new version

### Updating code
Normally you need to copy the `_search.py` file from `sklearn`. In there you need to set up the `RandomizedSearchCV` and `GridSearchCV` classes to accept the `spark` argument. 
SparkBaseSearchCV should also be created, and that needs to implement `fit()`.

### Updating version numbers
- Set the new scikit-spark version number in 
    - `setup.py`
    - `__init__.py` 
    - `changelog` (give description of work)
- Update `.github/workflows/pipeline.yml`
    - Add new version to test matrix

### Setting up tests
There is a bash script for running all tests `scikit-spark/python/run-tests.sh`.

See the `get_refactored_tests_to_skip` function. For each version it has a list of tests which need different handling to work correctly. Due to the way tests are added to `AllTests` it is not possible to use the `sklearn` method for parameterising the tests, so they have to be moved into `TestParameterisedTests`.  In most cases the original test can just be imported and called with the appropriate parameters.
