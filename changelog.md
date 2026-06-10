# 1.1.0
- scikit-learn 1.4–1.8 support added. Reworked `SparkBaseSearchCV.fit` to the sklearn 1.4+ metadata-routing API (SLEP006): `_check_fit_params` → `_check_method_params`, `fit(self, X, y=None, **params)` with `_get_routed_params_for_fit`, `score_params` threaded into `_fit_and_score`, and the `@_fit_context` decorator replacing the manual `_validate_params()`. `groups` is now passed through `routed_params.splitter.split`.
- Bumped `install_requires` to `scikit-learn~=1.8`; dev pins to `scikit-learn==1.8.0` / `pyspark==3.5.1` and numpy/scipy floors to the sklearn 1.8 minimums.

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
