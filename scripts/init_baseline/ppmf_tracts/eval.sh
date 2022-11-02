#!/bin/bash

# get $TRACTS
while getopts s: FLAG
do
    case "${FLAG}" in
        s) SEED=${OPTARG};;
    esac
done
source './scripts/ppmf_data_seeds.sh' $SEED

PROCESS=split

MARGINAL=-1
SUBSAMPLE=-1

NUM_MODELS=100
SYNTH=fixed
K=-1
T=1000

EVAL_METHOD=sample

for TRACT in "${TRACTS[@]}"
do
  DATASET=ppmf_$TRACT;
  echo $DATASET

  python eval.py --eval_gen_method $EVAL_METHOD \
    --dataset $DATASET --process_data $PROCESS \
    --marginal $MARGINAL --subsample_queries $SUBSAMPLE \
    --num_models $NUM_MODELS --synth $SYNTH --K $K --T $T \
    --init_split \
    --remove_existing_results;
done
