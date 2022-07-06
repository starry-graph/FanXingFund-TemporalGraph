#!/bin/bash

cd `dirname $0`
conda run -n wart python setup.py build_ext --inplace