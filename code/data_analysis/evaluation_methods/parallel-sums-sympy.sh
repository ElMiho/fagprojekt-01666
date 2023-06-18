#!/bin/bash

for i in $(seq 0 43); do
    python3 sums_sympy.py $i &
done