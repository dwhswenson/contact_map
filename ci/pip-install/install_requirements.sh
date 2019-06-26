#!/usr/bin/env bash

python -m pip install --upgrade pip
python -m pip install cython  # may be required for numpy override?
python -m pip install --upgrade --force-reinstall numpy  # override Travis numpy
python -m pip install -r requirements.txt
