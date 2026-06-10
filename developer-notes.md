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
- Update .travis.yml
    - Add `SKLEARN_VERSION` in test matrix
    - Add to excluded list for Python 2 if necessary

### Setting up tests
There is a bash script for running all tests `scikit-spark/python/run-tests.sh`.

## Running tests

Tasks are defined in `Taskfile.yml` ([go-task](https://taskfile.dev)).

Prerequisites:
- [`task`](https://taskfile.dev) to run the tasks
- [`act`](https://github.com/nektos/act) and Docker to run the CI pipeline locally

Commands:
- `task test:fast` — run the suite against your local Python/Spark environment (fast).
- `task test:act` — run the full GitHub Actions pipeline locally in Docker via `act`.
- `task test:act:fast` — faster iterative `act` run that reuses containers and skips the image pull. Non-hermetic: do a plain `task test:act` before trusting a green result.

The `act` runner image is pinned in `.actrc`.

See the `get_refactored_tests_to_skip` function. For each version it has a list of tests which need different handling to work correctly. Due to the way tests are added to `AllTests` it is not possible to use the `sklearn` method for parameterising the tests, so they have to be moved into `TestParameterisedTests`.  In most cases the original test can just be imported and called with the appropriate parameters.
