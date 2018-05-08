#!/bin/sh
rm -rf __pycache__
rm -rf build
rm -f fgm_eval.so
python setup.py build
cp build/lib.*/fgm_eval.*.so ./fgm_eval.so
