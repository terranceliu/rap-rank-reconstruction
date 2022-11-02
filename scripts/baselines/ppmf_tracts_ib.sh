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

for TRACT in "${TRACTS[@]}"
do
  DATASET=ppmf_$TRACT;
  echo $DATASET

  # tract/split/holdout
  python baseline.py --dataset $DATASET --process_data $PROCESS --ignore_block --remove_existing_results;

  # county
  DATASET_BASELINE=ppmf_${TRACT:0:5};
  echo $DATASET_BASELINE
  python baseline.py --dataset $DATASET --process_data $PROCESS --ignore_block --dataset_baseline $DATASET_BASELINE;

  # state
  DATASET_BASELINE=ppmf_${TRACT:0:2};
  echo $DATASET_BASELINE
  python baseline.py --dataset $DATASET --process_data $PROCESS --ignore_block --dataset_baseline $DATASET_BASELINE --save_counts;

  # national
  python baseline_ppmf_national.py --dataset $DATASET --process_data $PROCESS --ignore_block;
done
