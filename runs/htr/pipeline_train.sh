#!/bin/bash
# The running pipeline_sets3.sh execs this: the versioned training script.
exec env GPUS=1,2 bash packages/htr_train/lion_data_prep/train.sh
