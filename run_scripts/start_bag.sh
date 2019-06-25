#!/usr/bin/env bash

export PYTHONPATH=""

# disable QT session manager warnings
unset SESSION_MANAGER

exec ${BAG_PYTHON} -m IPython
