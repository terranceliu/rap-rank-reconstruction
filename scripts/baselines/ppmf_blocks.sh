#!/bin/bash

# get $TRACTS
source './scripts/ppmf_data_blocks.sh'

PROCESS=split

for TRACT in "${TRACTS[@]}"
do
  DATASET=ppmf_$TRACT;
  echo $DATASET

  # block/split/holdout
  python baseline.py --dataset $DATASET --process_data $PROCESS --ignore_block --remove_existing_results;

  # tract
  DATASET_BASELINE=ppmf_${TRACT:0:11};
  echo $DATASET_BASELINE
  python baseline.py --dataset $DATASET --process_data $PROCESS --ignore_block --dataset_baseline $DATASET_BASELINE;

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
