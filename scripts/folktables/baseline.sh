#!/bin/bash

YEAR=2018
STATES=(CA NY TX FL PA)
TASKS=(employment coverage mobility)
PROCESS=split

for STATE in "${STATES[@]}"
do
  for TASK in "${TASKS[@]}"
  do
    DATASET=folktables_${TASK}_${YEAR}_${STATE};
    echo $DATASET

    python baseline.py --dataset $DATASET --process_data $PROCESS --remove_existing_results;
  done
done
