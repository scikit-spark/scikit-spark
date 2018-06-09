# import sys
# if sys.version_info[:2] <= (2, 6):
#     try:
#         import unittest2 as unittest
#     except ImportError:
#     sys.stderr.write('Install unittest2 to test with Python 2.6 or earlier')
#         sys.exit(1)
# else:
#     import unittest

from pyspark.sql import SparkSession


def spark_test(cls):
    """
    Used as decorator to wrap around a class deriving from unittest.TestCase.
    Wraps current unittest methods setUpClass() and tearDownClass(), invoked by
    the nosetest command before and after unit tests are run.
    This enables us to create one PySpark SparkSession per test fixture.
    The session can be referred to with self.spark or ClassName.spark.

    The SparkSession is set up before invoking the class' own set up and torn
    down after the class' tear down, so you may safely refer to it in those
    methods.
    """
    setup = getattr(cls, 'setUpClass', None)
    teardown = getattr(cls, 'tearDownClass', None)

    def setUpClass(cls):
        cls.spark = create_local_spark_session("Unit Tests")
        if setup:
            setup()

    def tearDownClass(cls):
        if teardown:
            teardown()
        if cls.spark:
            cls.spark.stop()
            # Next session will attempt to reuse the previous stopped
            # SparkContext if it's not cleared.
            SparkSession._instantiatedContext = None
        cls.spark = None

    cls.setUpClass = classmethod(setUpClass)
    cls.tearDownClass = classmethod(tearDownClass)
    return cls


def create_local_spark_session(app_name="scikit-spark"):
    """
    Generates a :class:`SparkSession` utilizing all local cores
    with the progress bar disabled but otherwise default config.
    """
    return SparkSession.builder \
                       .master("local[*]") \
                       .appName(app_name) \
                       .config("spark.ui.showConsoleProgress", "false") \
                       .getOrCreate()
