#!/bin/bash

TEST_WAV_DIR="/vol/vssp/AP_datasets/audio/audioset/task4_dcase2017_audio/official_downloads/testing"
TRAIN_WAV_DIR="/vol/vssp/AP_datasets/audio/audioset/task4_dcase2017_audio/official_downloads/training"

WORKSPACE="/vol/vssp/msos/qk/workspaces/ICASSP2018_dcase"

# Extract features
# python prepare_data.py extract_features --wav_dir=$TEST_WAV_DIR --out_dir=$WORKSPACE"/features/logmel/testing" --recompute=True
python prepare_data.py extract_features --wav_dir=$TRAIN_WAV_DIR --out_dir=$WORKSPACE"/features/logmel/training" --recompute=True

#python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel/testing" --csv_path="meta_data/testing_set.csv" --out_path=$WORKSPACE"/packed_features/logmel/testing.h5"
#python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel/training" --csv_path="meta_data/training_set.csv" --out_path=$WORKSPACE"/packed_features/logmel/training.h5"
