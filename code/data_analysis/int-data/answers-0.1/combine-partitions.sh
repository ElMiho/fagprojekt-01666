#!/bin/bash

list=("0-2" "0-3" "0-4" "0-5" "0-6" "0-7" "0-8" "0-9" "0-10" "1-3" "1-4" "1-5" "1-6" "1-7" "1-8" "1-9" "1-10" "2-4" "2-5" "2-6" "2-7" "2-8" "2-9" "2-10" "3-5" "3-6" "3-7" "3-8" "3-9" "3-10" "4-6" "4-7" "4-8" "4-9" "4-10" "5-7" "5-8" "5-9" "5-10" "6-8" "6-9" "6-10" "7-9" "7-10")

for i in "${list[@]}"; do
    cat $(ls| grep $i) >> ../answers-0.1-no-partition/answers-$i.txt
done
