# COMP529 Project: Per-Data Parallel Group CPU Checkpointing for BERT

This README contains instructions to reproduce results/experiments provided in the final report.

## Changes

The primary changes all live in the `run_pretraining.py` file. We've provided a simple script, `kill_processes.sh` to help you kill processes in between experiment (each expirement creates 10s of processes that need to be deleted cleanly in order to move to the next experiment). `split_array_into_equal_parts.py` is a script which trivially produces the splits for the model and optimizer states, but it can be useful if you want to inspect the size of the tensors in these dictionaries.

## Data Downloading and Processing

BERT training using the Wikipedia dataset. Instructions to download this dataset are provided [here](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/README.md#getting-the-data). Specifically, you need to run `bash data/create_datasets_from_start.sh`. Please note this process can take a few hours as the dataset size is quite large (10s of GBs).

## Training and Taking CPU Snapshots

To run the code and have it take CPU snapshots, no additional configuration is needed. All commands assume you're currently in this folder: `cd PyTorch/LanguageModeling/BERT`

Simply run: `bash scripts/run_pretraining.sh $(source scripts/configs/pretrain_config.sh && dgxa100-80g_8gpu_fp16)`.

This process will use 8 GPUs and thus 8 data parallel groups. To adjust the number of data parallel groups, a few manual steps are needed:
1. In the file `scripts/configs/pretrain_config.sh`, under the `dgxa100-80g_8gpu_fp16` configuration, you'll need to adjust `num_gpus` to the number you want to train on. For now, it can only be in the set {1, 2, 4, 8}
2. In run_pretraining.py, you'll need to copy-replace instances of `model_state_split_indices_8` and `optimizer_state_split_indices_8` to `model_state_split_indices_{num_gpus}` and `optimizer_state_split_indices_{num_gpus}`, where `num_gpus` is the value you set earlier. This is to ensure the model splitting is correct.


## Persist CPU Checkpoint to Disk

1. First make sure there is no directory called `memory` in `results/checkpoints`. If there is, please delete it. Then create a fresh copy of this directory so disk checkpoints can be written here: `mkdir memory; mkdir memory/grad_scaler; mkdir memory/optimizer_misc`
2. Ensure there is no file called `error.txt`. Start training (CPU checkpoints are enabled by default). In another terminal window, create the `error.txt file`: `touch error.txt`

This will stop the current training and start the CPU checkpoints to be persisted to disk.


## Resume Training from Checkpoint on Disk

To resume training from the latest checkpoint on disk (located in the `results/checkpoints/memory` folder), set `resume_from_in_mem_checkpoints` in `run_pretraining.py` to `True`. This will invoke the program to pull the load the latest checkpoint in all trainers and resume training. Also make sure to delete the `error.txt` file created in the prior step or the trainers will notice this file and immediately try to checkpoint to disk