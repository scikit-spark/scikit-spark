# 1.0.0
- Dropped support for scikit-learn 0.x (0.19–0.22)
- Added support for scikit-learn 1.1 through 1.8 for GridSearchCV and RandomizedSearchCV (per-minor vendored `_search.py` under `skspark/sklearn_1_<minor>`, selected at runtime by installed version)
- Migrated CI from Travis to GitHub Actions with a Python / PySpark / Java / scikit-learn test matrix
- Switched the test runner from nose to pytest
- Removed the dead nose-era `python/run-tests.sh`
- Dropped testing for Java 8 and Python 3.8 / 3.9

# 0.4.0
- scikit-learn 0.22 support added

# 0.3.0

- scikit-learn 0.21 support added
- scikit-learn 0.21 dropped support for Python 2 and that is mirrored in scikit-spark

# 0.2.0

- scikit-learn 0.20 support added
- Automated testing with travis setup

# 0.1.0

- Tested with real data
- Fixed bug where you couldn't pickle an RS or GS object more than once
    - https://github.com/scikit-spark/scikit-spark/issues/6
- Made RS and GS work without Spark and added a test     
- Made the `spark` argument work with boolean values
- Added `spark` argument to the docstrings 

# 0.1.0rc4

- Tests run in Python 2 and Python 3 now
- Fixed README example (not enough parameters for RS to try)


# 0.1.0rc2

- Fixed setup.py import for skspark


# 0.1.0rc1

- Done
    - Fully tested and working `GridSearchCV` and `RandomizedSearchCV` (python2)
    - A basic start to the `README.md` has been made
    - Project has been uploaded to PyPi

- Essential to still come
    - Filling out the rest of the `README.md`
    - Docstrings for `GridSearchCV` and `RandomizedSearchCV`
    - Python 3 testing
    - Setting up travis
