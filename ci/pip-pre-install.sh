#!/usr/bin/env bash

pip install --upgrade pip
pip install cython  # may be required for numpy override?
pip install --upgrade --force-reinstall numpy  # override Travis numpy
#pip install -r requirements.txt
