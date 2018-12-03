#!/bin/bash
# [.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
for percTrain in `seq 0.01 0.01 .1`; do
    echo $percTrain
    python HeadleyJonathon-KLR.py $percTrain .5 | tail -n1 | tee -a output.csv
done
