#!/bin/bash

inputDataset="pan12-authorship-attribution-test-dataset-problem-a-2015-10-20/"
output="output/"

. init_environment.sh && python3 raghavan10.py -i $inputDataset -o $output

octave -q authorship_attribution_eval.m -a $output/answers.json -t $inputDataset/ground-truth.json -m $inputDataset/meta-file.json -o $output/evaluation.prototext
