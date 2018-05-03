#!/bin/sh
rm -rf build
rm fgm_eval.c
python setup.py build
cp build/lib.macosx-10.7-x86_64-3.6/fgm_eval.cpython-36m-darwin.so ./fgm_eval.so
echo "DONE!"
