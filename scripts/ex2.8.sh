#!/usr/bin/env bash
echo 'KNN'
for k in 1 3 5 7 15; do
    out=$(python3 KNN/KNN_classify.py $k)
    echo "$k $out"
done
echo 'Least squares'
python3 linear_regression/least_squares.py

