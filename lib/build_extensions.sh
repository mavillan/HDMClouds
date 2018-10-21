#!/bin/sh
rm -rf __pycache__
rm -rf build
rm -f fgm_eval.so
rm -f gmr.so
python setup_fgm.py build
python setup_gmr.py build
cp build/lib.*/fgm_eval.*.so ./fgm_eval.so
cp build/lib.*/gmr.*.so ./gmr.so
