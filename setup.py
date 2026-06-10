from setuptools import find_packages, setup

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
    # Per-minor support is vendored under skspark/sklearn_1_<minor>; the exact
    # version is pinned per CI matrix row. See developer-notes.md.
    "scikit-learn>=1.1,<1.9",
    # numpy<2: sklearn 1.1-1.3 wheels are built against the numpy 1.x ABI and
    # crash on numpy 2; numpy 1.26 is runtime-compatible across the 1.1-1.8 range.
    "numpy>=1.22,<2",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="scikit-spark",
    version="1.0.0",
    author="Ganesh N. Sivalingam",
    author_email="g.n.sivalingam@gmail.com",
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=keywords,
    package_dir={"": "python"},
    packages=find_packages("python", exclude="tests"),
    url="https://github.com/scikit-spark/scikit-spark",
    install_requires=install_requires,
    extras_require={
        "spark": ["pyspark[sql]~=3.0"],
    },
    license="Apache 2.0"
)

# TODO add classifiers
