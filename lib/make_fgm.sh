#!/bin/sh
rm -rf build
rm fgm_eval.c
python setup.py build
cp build/lib.*/fgm_eval.*.so ./fgm_eval.so
echo "DONE!"
