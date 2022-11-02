#!/bin/bash

YEAR=2018
STATES=(CA NY TX FL PA)
TASKS=(employment coverage mobility)
PROCESS=split

SUBSAMPLE=-1

NUM_MODELS=100
SYNTH=fixed
K=1000
T=1000

EVAL_METHOD=sample
NUM_SAMPLES=50000

for MARGINAL in 2 3
do
  for STATE in "${STATES[@]}"
  do
    for TASK in "${TASKS[@]}"
    do
      DATASET=folktables_${TASK}_${YEAR}_${STATE};
      python eval.py --eval_gen_method $EVAL_METHOD \
        --dataset $DATASET --process_data $PROCESS \
        --marginal $MARGINAL --subsample_queries $SUBSAMPLE \
        --num_models $NUM_MODELS --synth $SYNTH --K $K --T $T \
        --num_samples $NUM_SAMPLES \
        --remove_existing_results;
    done
  done
done