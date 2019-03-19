#!/usr/bin/env bash

pip install --upgrade pip
pip install cython  # may be required for numpy override?
pip install --upgrade --force-reinstall numpy  # override Travis numpy
# --no-binary required until MDTraj updates its wheels
pip install --no-binary -r requirements.txt
