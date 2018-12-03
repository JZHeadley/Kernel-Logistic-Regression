#!/bin/bash
for percTrain in `seq 0.01 0.01 .07`; do
    echo $percTrain
    python HeadleyJonathon-SKLR.py $percTrain .5 | tail -n1 | tee -a output.csv
done
