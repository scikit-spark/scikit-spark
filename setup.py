from setuptools import setup, find_packages

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

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="scikit-spark",
    version="0.1.0rc1",
    author="Ganesh N. Sivalingam",
    author_email="g.n.sivalingam@gmail.com",
    description=description,
    long_description=long_description,
    keywords=keywords,
    package_dir={"": "python"},
    packages=find_packages("python", exclude="tests"),
    url="https://github.com/scikit-spark/scikit-spark",
    install_requires=install_requires,
    license="Apache 2.0"
)

# TODO add classifiers
