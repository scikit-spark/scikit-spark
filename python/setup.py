from setuptools import setup

description = "Spark acceleration for Scikit-Learn cross validation techniques"

keywords = [
    "spark",
    "pyspark",
    "scikit-learn",
    "sklearn",
    "machine learning",
    "random search",
    "grid search"
]

install_requires = [
    "scikit-learn>=0.19",
    "six==1.11.0"
]

setup(
    name="scikit-spark",
    version="0.1.0rc1",
    author="Ganesh N. Sivalingam",
    author_email="g.n.sivalingam@gmail.com",
    description=description,
    keywords=keywords,
    packages=["skspark"],
    url="https://github.com/scikit-spark/scikit-spark",
    install_requires=install_requires,
    licence="Apache 2.0"
)

# TODO add classifiers
# TODO add long_description (README.md)
