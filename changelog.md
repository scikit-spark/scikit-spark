# 1.0.0
- Scikit-learn support
  - added >=1.2 support
  - dropped support for 0.*, use previous versions for support
- Spark version support
  - added support for Spark 3.3.2 and 3.4.0
  - dropped support for Spark 2.*, use previous versions for support
- Java 11 support and testing added (in addition to Java 8)
- Automated testing migrated to Github Actions from Travis

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
