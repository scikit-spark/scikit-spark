#!/usr/bin/env bash
# Runs only the doctests. Additional flags are passed through to nose.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${DIR}/env_setup.sh

ALL_MODULES=$(cd $DIR && \
    echo 'import skspark, inspect; \
    print(" ".join("skspark." + x[0] \
    for x in inspect.getmembers(skspark, inspect.ismodule)))' \
    | python)

${DIR}/run-tests.sh $ALL_MODULES --with-doctest $@
