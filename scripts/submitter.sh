#!/bin/bash
#PBS -l cput=2000:00:00
#PBS -l walltime=2000:00:00

source /user/m/marvill/miniconda3/bin/activate base
python /user/m/marvill/HDMClouds/scripts/$1 $2 
