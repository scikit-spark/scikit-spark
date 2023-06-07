#!/usr/bin/env bash
# Runs both doctests and unit tests by default, otherwise hands arguments over
# to nose.

# The current directory of the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ "$#" = 0 ]]; then
    ARGS="--nologcapture --all-modules --verbose"
else
    ARGS="$@"
fi

export PYSPARK_PYTHON=$(which python)
pytest --disable-warnings -vv -rxXs $DIR/test
