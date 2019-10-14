import unittest
import logging
from pyspark.sql import SparkSession


class PySparkTest(unittest.TestCase):
    """
    Based on this blog post:
    https://blog.cambridgespark.com/unit-testing-with-pyspark-fb31671b1ad8
    """
    @staticmethod
    def suppress_py4j_logging():
        logger = logging.getLogger("py4j")
        logger.setLevel(logging.WARN)

    @staticmethod
    def create_testing_pyspark_session():
        return SparkSession.builder\
                .master("local[*]")\
                .appName("scikit-spark-tests")\
                .getOrCreate()

    @classmethod
    def setUpClass(cls):
        cls.suppress_py4j_logging()
        cls.spark = cls.create_testing_pyspark_session()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
