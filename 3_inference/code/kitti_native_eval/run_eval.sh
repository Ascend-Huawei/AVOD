#!/bin/bash

set -e

cd $1
echo "200000" | tee -a $5/results_AP_$2.txt
./evaluate_object_3d_offline $3 $4 | tee -a $5/results_AP_$2.txt
