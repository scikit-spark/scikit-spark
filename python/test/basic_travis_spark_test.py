"""
This just checks if spark has been installed correctly
"""
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([
    (1, 2),
    (2, 3),
], ["a", "b"])

df.show()
