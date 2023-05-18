import logging

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    spark_session = SparkSession.builder\
        .master("local[*]")\
        .appName("scikit-spark-tests")\
        .getOrCreate()

    logger = logging.getLogger("py4j")
    logger.setLevel(logging.WARN)

    return spark_session
