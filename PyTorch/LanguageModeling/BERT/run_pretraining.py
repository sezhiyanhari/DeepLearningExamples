# coding=utf-8
# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
import os
import time
import argparse
import random
import logging
import h5py
from tqdm import tqdm, trange
from typing import Final, Any, Callable
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math

import modeling
from schedulers import PolyWarmUpScheduler
from lamb_amp_opt.fused_lamb import FusedLAMBAMP

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utils import is_main_process, format_step, get_world_size, get_rank
from torch.nn.parallel import DistributedDataParallel as DDP
from schedulers import LinearWarmUpScheduler

import dllogger

import lddl.torch

import time
import copy
import json

# Enabling the TorchScript Runtime Backend NVFuser
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)

# Track whether a SIGTERM (cluster time up) has been handled
timeout_sent = False

import signal
# handle SIGTERM sent from the scheduler and mark so we
# can gracefully save & exit
def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True

signal.signal(signal.SIGTERM, signal_handler)

optimizer_state_split_indices_8 = [0, 16, 58, 99, 138, 182, 221, 266, 796]
optimizer_state_split_indices_6 = [0, 29, 86, 137, 194, 245, 796]
optimizer_state_split_indices_4 = [0, 57, 137, 220, 796]
optimizer_state_split_indices_2 = [0, 137, 796]
optimizer_state_split_indices_1 = [0, 796]

model_state_split_indices_8 = [0, 23, 81, 143, 203, 257, 319, 379, 399]
model_state_split_indices_6 = [0, 47, 127, 207, 287, 367, 399]
model_state_split_indices_4 = [0, 81, 203, 319, 399]
model_state_split_indices_2 = [0, 197, 399]
model_state_split_indices_1 = [0, 399]

class BertPretrainingCriterion(torch.nn.Module):

    sequence_output_is_dense: Final[bool]

    def __init__(self, vocab_size, sequence_output_is_dense=False):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        self.sequence_output_is_dense = sequence_output_is_dense

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        if self.sequence_output_is_dense:
            # prediction_scores are already dense
            masked_lm_labels_flat = masked_lm_labels.view(-1)
            mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -1]
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


class SyncFreeStats :
    def __init__(self) :
        self.host_stats = {}
        self.device_stats = {}
        self.device_funcs = {}

    def add_stat(self, name, dtype=torch.int32, device_tensor=None, device_func=None) :
        if device_tensor is not None :
            assert dtype == device_tensor.dtype, "Error: dtype do not match: {} {}".format(dtype, device_tensor.dtype)
        self.host_stats[name] = torch.zeros(1, dtype=dtype).pin_memory()
        self.device_stats[name] = device_tensor
        self.device_funcs[name] = device_func

    def copy_from_device(self) :
        for name in self.host_stats.keys() :
            # Apply device function to device stat
            if self.device_stats[name] is not None and self.device_funcs[name] is not None:
                self.host_stats[name].copy_(self.device_funcs[name](self.device_stats[name]), non_blocking=True)
            elif self.device_stats[name] is not None :
                self.host_stats[name].copy_(self.device_stats[name], non_blocking=True)
            elif self.device_funcs[name] is not None :
                self.host_stats[name].copy_(self.device_funcs[name](), non_blocking=True)

    def host_stat(self, name) :
        assert name in self.host_stats
        return self.host_stats[name]

    def host_stat_value(self, name) :
        assert name in self.host_stats
        return self.host_stats[name].item()

    def update_host_stat(self, name, tensor) :
        self.host_stats[name] = tensor

    def device_stat(self, name) :
        assert self.device_stats[name] is not None
        return self.device_stats[name]

    def update_device_stat(self, name, tensor) :
        self.device_stats[name] = tensor


def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .parquet files for the task.")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument('--vocab_file',
                        type=str,
                        default=None,
                        required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--resume_phase2',
                        default=False,
                        action='store_true',
                        help="Whether to resume training with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument('--init_loss_scale',
                        type=int,
                        default=2**20,
                        help="Initial loss scaler value")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--json-summary', type=str, default="results/dllogger.json",
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument('--steps_this_run', type=int, default=-1,
                        help='If provided, only run this many steps before exiting')
    parser.add_argument("--profile",
                        default=False,
                        action='store_true',
                        help="Whether to profile model.")
    parser.add_argument("--profile-start",
                        default=0,
                        type=int,
                        help="Delay profiling to start step.")
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of DataLoader worker processes per rank')
    # optimizations controlled by command line arguments
    parser.add_argument("--no_dense_sequence_output",
                        default=False,
                        action='store_true',
                        help="Disable dense sequence output")
    parser.add_argument("--disable_jit_fusions",
                        default=False,
                        action='store_true',
                        help="Disable jit fusions.")
    parser.add_argument("--cuda_graphs",
                        default=False,
                        action='store_true',
                        help="Enable Cuda Graphs.")

    args = parser.parse_args()
    args.fp16 = args.fp16 or args.amp

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    return args

def setup_training(args):

    assert (torch.cuda.is_available())

    if args.local_rank == -1:
        device = torch.device("cuda", 0)
        args.n_gpu = 1 # torch.cuda.device_count()
        args.allreduce_post_accumulation = False
        args.allreduce_post_accumulation_fp16 = False
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        if args.cuda_graphs :
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.n_gpu = 1

    if is_main_process():
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    dllogger.metadata("e2e_train_time", {"unit": "s"})
    dllogger.metadata("training_sequences_per_second", {"unit": "sequences/s"})
    dllogger.metadata("final_loss", {"unit": None})
    dllogger.metadata("raw_train_time", {"unit": "s"})

    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def prepare_model_and_optimizer(args, device, buffer_states, model_state_keys, optimizer_state_keys, sequence_output_is_dense):

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    model = modeling.BertForPreTraining(config, sequence_output_is_dense=sequence_output_is_dense)

    resume_from_in_mem_checkpoints = False

    checkpoint = None
    if resume_from_in_mem_checkpoints:
        args.resume_step = 10
        global_step = 0
        start_time = time.time()
        load_model_checkpoints_from_disk_optimized(args, model, model_state_keys)
        end_time = time.time()
        print(f"Total time to load model from checkpoint on disk backed by our in-mem checkpointing: {end_time - start_time}")
    elif not args.resume_from_checkpoint:
        global_step = 0
    else:
        start_time = time.time()
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step if not args.init_checkpoint else 0

        if not args.init_checkpoint:
            checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location=device)
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model'], strict=False)

        if args.phase2 and not args.init_checkpoint:
            global_step -= args.phase1_end_step
        if args.init_checkpoint:
            args.resume_step = 0
        if is_main_process():
            print("resume step from ", args.resume_step)
        end_time = time.time()
        print(f"Total time to load model checkpoint from disk using baseline checkpointing system: {end_time - start_time}")

    model.to(device)

    # If allreduce_post_accumulation_fp16 is not set, Native AMP Autocast is
    # used along with FP32 gradient accumulation and all-reduce
    if args.fp16 and args.allreduce_post_accumulation_fp16:
        model.half()

    if not args.disable_jit_fusions :
        model = torch.jit.script(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=args.warmup_proportion,
                                       total_steps=args.max_steps,
                                       base_lr=args.learning_rate,
                                       device=device)
    grad_scaler = torch.cuda.amp.GradScaler(init_scale=args.init_loss_scale, enabled=args.fp16)

    model.checkpoint_activations(args.checkpoint_activations)

    if resume_from_in_mem_checkpoints:
        start_time = time.time()
        load_optimizer_checkpoints_from_disk_optimized(args, optimizer, optimizer_state_keys, device)
        load_grad_scaler_checkpoints_from_disk(grad_scaler)
        end_time = time.time()
        print(f"Total time to load optimizer/grad_scaler checkpoint on disk backed by our in-mem checkpointing: {end_time - start_time}")
    else:
        start_time = time.time()
        if args.resume_from_checkpoint:
            if (args.phase2 and not args.resume_phase2) or args.init_checkpoint :
                for group in checkpoint['optimizer']['param_groups'] :
                    group['step'].zero_()
                    group['lr'].fill_(args.learning_rate)
            else :
                if 'grad_scaler' in checkpoint and (not args.phase2 or args.resume_phase2):
                    grad_scaler.load_state_dict(checkpoint['grad_scaler'])
            optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)
        end_time = time.time()
        print(f"Total time to load optimizer/grad_scaler checkpoint on original checkpointing system: {end_time - start_time}")

    if args.local_rank != -1:
        # Cuda Graphs requires that DDP is captured on a side stream
        # It is important to synchronize the streams after the DDP initialization
        # so anything after sees properly initialized model weights across GPUs
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream) :
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory, gradient_as_bucket_view=True)
        torch.cuda.current_stream().wait_stream(side_stream)

        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
        def scale_by_grad_accum_steps_wrapper(hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:

            def scale_by_grad_accum_steps_wrapper_hook(
                hook_state, bucket: dist.GradBucket
            ) -> torch.futures.Future[torch.Tensor]:
                bucket.set_buffer(bucket.buffer().div_(args.gradient_accumulation_steps))
                fut = hook(hook_state, bucket)
                return fut

            return scale_by_grad_accum_steps_wrapper_hook

        # With gradient accumulation, the DDP comm hook divides the gradients by the number
        # gradient accumulation steps
        if args.gradient_accumulation_steps > 1:
            model.register_comm_hook(None, scale_by_grad_accum_steps_wrapper(allreduce_hook))

    optimizer.setup_fp32_params()

    criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=sequence_output_is_dense)

    if (args.resume_from_checkpoint and not args.phase2) or (args.resume_phase2) or args.init_checkpoint:
        start_epoch = checkpoint.get('epoch', 0)
    else:
        start_epoch = 0

    return model, optimizer, grad_scaler, lr_scheduler, checkpoint, global_step, criterion, start_epoch


def checkpoint_step(args, epoch, global_step, model, optimizer, grad_scaler, last3_checkpoint_paths):
    torch.cuda.synchronize()
    if is_main_process() and not args.skip_checkpoint:
        # Save a trained model
        start_time = time.time()
        dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        if args.resume_step < 0 or not args.phase2:
            output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
        else:
            output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step + args.phase1_end_step))
        if args.do_train:
            torch.save({'model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'grad_scaler': grad_scaler.state_dict(),
                        'epoch': epoch}, output_save_file)

            # The new checkpoint could have a name already in
            # last3_checkpoint_paths. In this case, torch.save will overwrite
            # the old file; thus, we need to take the name out of
            # last3_checkpoint_paths and append it to the last.
            if output_save_file in last3_checkpoint_paths:
                last3_checkpoint_paths.remove(output_save_file)
            last3_checkpoint_paths.append(output_save_file)
            if len(last3_checkpoint_paths) > 3:
                ckpt_to_be_removed = last3_checkpoint_paths.pop(0)
                os.remove(ckpt_to_be_removed)
        end_time = time.time()
        print(f"Total time to checkpoint all state: {end_time - start_time}")


def write_cpu_checkpoint_to_disk(args, buffer_states, grad_scaler, model_state_keys, optimizer_state_keys):
    print("Error detected, writing in-memory checkpoints to disk...")

    start_time = time.time()

    if args.local_rank == 0:
        with open('results/checkpoints/memory/grad_scaler/grad_scaler.txt', 'w+') as grad_scaler_file: 
            grad_scaler_file.write(json.dumps(grad_scaler.state_dict()))
        
        with open('results/checkpoints/memory/optimizer_misc/optimizer_state_param_groups.txt', 'w+') as optimizer_state_param_groups_file:
            optimizer_state_param_groups_file.write(json.dumps(buffer_states['optimizer_state_param_groups']))
        
        with open('results/checkpoints/memory/optimizer_misc/optimizer_state_tensor_params.txt', 'w+') as optimize_state_tensor_params_file:
            optimize_state_tensor_params_file.write(json.dumps(buffer_states['optimizer_state_tensor_params']))

    # for model_state_counter in range(model_state_split_indices_8[args.local_rank], model_state_split_indices_8[args.local_rank + 1]):
    #     k = model_state_keys[model_state_counter]
    #     buffer_state_key = f"{k}"
    #     torch.save(buffer_states[buffer_state_key], f"results/checkpoints/memory/model_state/{buffer_state_key}.pt")

    # for optimizer_state_counter in range(optimizer_state_split_indices_8[args.local_rank], optimizer_state_split_indices_8[args.local_rank + 1]):
    #     state_key, k = optimizer_state_keys[optimizer_state_counter]
    #     buffer_state_key = f"{state_key}-{k}"
    #     torch.save(buffer_states[buffer_state_key], f"results/checkpoints/memory/optimizer_state/{buffer_state_key}")

    # Faster way to store entire state to disk instead of individual files
    torch.save(buffer_states, f"results/checkpoints/memory/buffer_states_{args.local_rank}.pt")    

    end_time = time.time()

    print(f"Done writing checkpoints to disk, took {end_time - start_time} seconds...")

    exit(0)


def load_model_checkpoints_from_disk(model, model_state_keys):
    print("Loading model state from latest checkpoint...")
    state_dict = {}
    for k in model_state_keys:
        buffer_state_key = f"{k}"
        state_dict[k] = torch.load(f"results/checkpoints/memory/model_state/{buffer_state_key}.pt")
    model.load_state_dict(state_dict)
    print("Finished loading model state")


def load_model_checkpoints_from_disk_optimized(args, model, model_state_keys):
    print(f"Loading model state from latest checkpoint optimized {args.n_gpu}...")
    state_dict = {}
    for local_rank in range(8):
        buffer_states = torch.load(f"results/checkpoints/memory/buffer_states_{local_rank}.pt")
        for k in model_state_keys[model_state_split_indices_8[local_rank]: model_state_split_indices_8[local_rank + 1]]:
            state_dict[k] = buffer_states[k]
    model.load_state_dict(state_dict)


def load_optimizer_checkpoints_from_disk(optimizer, optimizer_state_keys, device):
    print("Loading optimizer state from latest checkpoint...")
    state_dict = {}
    state_dict['state'] = {}
    for param_num, param_attr in optimizer_state_keys:
        if param_num not in state_dict['state']:
            state_dict['state'][param_num] = {}
        state_dict['state'][param_num][param_attr] = torch.load(f"results/checkpoints/memory/optimizer_state/{param_num}-{param_attr}").to(device)
    
    with open('results/checkpoints/memory/optimizer_misc/optimizer_state_param_groups.txt', 'r') as optimizer_state_param_groups_file:
        optimizer_state_param_groups_from_file = json.load(optimizer_state_param_groups_file)        
        state_dict['param_groups'] = optimizer_state_param_groups_from_file

        with open('results/checkpoints/memory/optimizer_misc/optimizer_state_tensor_params.txt', 'r') as optimize_state_tensor_params_file:
            optimize_state_tensor_params_from_file = json.load(optimize_state_tensor_params_file)
            for param_group in state_dict['param_groups']:
                for k in param_group:
                    if k in optimize_state_tensor_params_from_file:
                        v = param_group[k]
                        int_dtype = True if "Int" in optimize_state_tensor_params_from_file[k] else False
                        if int_dtype:
                            param_group[k] = torch.IntTensor([v]).to(device) #dtype=optimize_state_tensor_params_from_file[k])
                        else:
                            param_group[k] = torch.Tensor([v]).to(device)
    optimizer.load_state_dict(state_dict)
    print("Finished loading optimizer state")


def load_optimizer_checkpoints_from_disk_optimized(args, optimizer, optimizer_state_keys, device):
    print("Loading optimizer state from latest checkpoint optimized...")
    state_dict = {}
    state_dict['state'] = {}
    for local_rank in range(8):
        buffer_states = torch.load(f"results/checkpoints/memory/buffer_states_{local_rank}.pt")
        for param_num, param_attr in optimizer_state_keys[optimizer_state_split_indices_8[local_rank]: optimizer_state_split_indices_8[local_rank + 1]]:
            if param_num not in state_dict['state']:
                state_dict['state'][param_num] = {}
            state_dict['state'][param_num][param_attr] = buffer_states[f"{param_num}-{param_attr}"].to(device)
    
    with open('results/checkpoints/memory/optimizer_misc/optimizer_state_param_groups.txt', 'r') as optimizer_state_param_groups_file:
        optimizer_state_param_groups_from_file = json.load(optimizer_state_param_groups_file)        
        state_dict['param_groups'] = optimizer_state_param_groups_from_file

        with open('results/checkpoints/memory/optimizer_misc/optimizer_state_tensor_params.txt', 'r') as optimize_state_tensor_params_file:
            optimize_state_tensor_params_from_file = json.load(optimize_state_tensor_params_file)
            for param_group in state_dict['param_groups']:
                for k in param_group:
                    if k in optimize_state_tensor_params_from_file:
                        v = param_group[k]
                        int_dtype = True if "Int" in optimize_state_tensor_params_from_file[k] else False
                        if int_dtype:
                            param_group[k] = torch.IntTensor([v]).to(device) #dtype=optimize_state_tensor_params_from_file[k])
                        else:
                            param_group[k] = torch.Tensor([v]).to(device)
    optimizer.load_state_dict(state_dict)
    print("Finished loading optimizer state")


def load_grad_scaler_checkpoints_from_disk(grad_scaler):
    print("Loading grad scaler from latest checkpoint...")
    with open('results/checkpoints/memory/grad_scaler/grad_scaler.txt', 'r') as grad_scaler_file: 
        grad_scaler_from_file = json.load(grad_scaler_file)
        grad_scaler.load_state_dict(grad_scaler_from_file)
        print("Loaded grad_scaler: ", grad_scaler)


def checkpoint_only_to_cpu_step(args, epoch, global_step, model, optimizer, grad_scaler, last3_checkpoint_paths, buffer_states, model_state_keys, optimizer_state_keys):
    torch.cuda.synchronize(device=args.local_rank)
    start_checkpoint_time = time.time()
    dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
    model_to_save = model.module if hasattr(model,
                                            'module') else model  # Only save the model it-self
    if args.do_train:
        model_state_full = model_to_save.state_dict()
        optimizer_state_full = optimizer.state_dict()['state']

        for model_state_counter in range(model_state_split_indices_8[args.local_rank], model_state_split_indices_8[args.local_rank + 1]):
            k = model_state_keys[model_state_counter]
            buffer_state_key = f"{k}"
            buffer_states[buffer_state_key].copy_(model_state_full[k], non_blocking=True)

        for optimizer_state_counter in range(optimizer_state_split_indices_8[args.local_rank], optimizer_state_split_indices_8[args.local_rank + 1]):
            state_key, k = optimizer_state_keys[optimizer_state_counter]
            buffer_state_key = f"{state_key}-{k}"
            buffer_states[buffer_state_key].copy_(optimizer_state_full[state_key][k], non_blocking=True)

        if args.local_rank == 0:
            buffer_states["optimizer_state_param_groups"] = []
            buffer_states["optimizer_state_tensor_params"] = {}
            param_groups = optimizer.state_dict()['param_groups']
            for param_group in param_groups:
                param_group_copy = {}
                for k in param_group:
                    v = param_group[k]
                    if isinstance(v, torch.Tensor):
                        param_group_copy[k] = v.item()
                        buffer_states["optimizer_state_tensor_params"][k] = v.type()
                    else:
                        param_group_copy[k] = v
                buffer_states["optimizer_state_param_groups"].append(param_group_copy)


    torch.cuda.synchronize(device=args.local_rank)
    end_checkpoint_time = time.time()
    print(f"Total time to checkpoint to CPU for rank {args.local_rank}: {end_checkpoint_time - start_checkpoint_time}")


def create_cpu_buffers_for_checkpointing(args, epoch, global_step, model, optimizer, grad_scaler, last3_checkpoint_paths, buffer_states, model_state_keys, optimizer_state_keys):
    print("Creating CPU buffers...")
    torch.cuda.synchronize(device=args.local_rank)

    model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self

    model_state_counter = 0
    model_state_dict_cpu = model_to_save.state_dict()
    for k, v in model_state_dict_cpu.items():
        buffer_state_key = f"{k}"
        if model_state_counter >= model_state_split_indices_8[args.local_rank] and model_state_counter < model_state_split_indices_8[args.local_rank + 1]:
            buffer_states[buffer_state_key] = torch.randn(v.shape, device='cpu', pin_memory=True)
        model_state_counter += 1

    optimizer_state_counter = 0
    optimizer_state_dict_cpu = optimizer.state_dict()
    for state_key, state in optimizer_state_dict_cpu['state'].items():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                buffer_state_key = f"{state_key}-{k}"
                if optimizer_state_counter >= optimizer_state_split_indices_8[args.local_rank] and optimizer_state_counter < optimizer_state_split_indices_8[args.local_rank + 1]:
                    buffer_states[buffer_state_key] = torch.randn(v.shape, device='cpu', pin_memory=True)
                optimizer_state_counter += 1

    torch.cuda.synchronize(device=args.local_rank)


def take_training_step(args, grad_scaler, model, criterion, batch, stats):
    with torch.cuda.amp.autocast(enabled=(args.fp16 and not args.allreduce_post_accumulation_fp16)) :
        prediction_scores, seq_relationship_score = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], masked_lm_labels=batch['labels'])
        loss = criterion(prediction_scores, seq_relationship_score, batch['labels'], batch['next_sentence_labels'])

    stats.device_stat('average_loss').add_(loss.detach())
    grad_scaler.scale(loss).backward()


def take_optimizer_step(args, lr_scheduler, optimizer, grad_scaler, device, stats):
    lr_scheduler.step()  # learning rate warmup
    grad_scaler.step(optimizer)

    # Stats copying is located here prior to the infinity check being reset
    # in GradScaler::update()
    stats.copy_from_device()

    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)


def main():
    global timeout_sent

    args = parse_arguments()

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)

    device, args = setup_training(args)
    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    buffer_states = {}
    model_state_keys = ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight', 'bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.bias', 'bert.encoder.layer.0.attention.self.query.weight', 'bert.encoder.layer.0.attention.self.query.bias', 'bert.encoder.layer.0.attention.self.key.weight', 'bert.encoder.layer.0.attention.self.key.bias', 'bert.encoder.layer.0.attention.self.value.weight', 'bert.encoder.layer.0.attention.self.value.bias', 'bert.encoder.layer.0.attention.output.dense.weight', 'bert.encoder.layer.0.attention.output.dense.bias', 'bert.encoder.layer.0.attention.output.LayerNorm.weight', 'bert.encoder.layer.0.attention.output.LayerNorm.bias', 'bert.encoder.layer.0.intermediate.dense_act.weight', 'bert.encoder.layer.0.intermediate.dense_act.bias', 'bert.encoder.layer.0.output.dense.weight', 'bert.encoder.layer.0.output.dense.bias', 'bert.encoder.layer.0.output.LayerNorm.weight', 'bert.encoder.layer.0.output.LayerNorm.bias', 'bert.encoder.layer.1.attention.self.query.weight', 'bert.encoder.layer.1.attention.self.query.bias', 'bert.encoder.layer.1.attention.self.key.weight', 'bert.encoder.layer.1.attention.self.key.bias', 'bert.encoder.layer.1.attention.self.value.weight', 'bert.encoder.layer.1.attention.self.value.bias', 'bert.encoder.layer.1.attention.output.dense.weight', 'bert.encoder.layer.1.attention.output.dense.bias', 'bert.encoder.layer.1.attention.output.LayerNorm.weight', 'bert.encoder.layer.1.attention.output.LayerNorm.bias', 'bert.encoder.layer.1.intermediate.dense_act.weight', 'bert.encoder.layer.1.intermediate.dense_act.bias', 'bert.encoder.layer.1.output.dense.weight', 'bert.encoder.layer.1.output.dense.bias', 'bert.encoder.layer.1.output.LayerNorm.weight', 'bert.encoder.layer.1.output.LayerNorm.bias', 'bert.encoder.layer.2.attention.self.query.weight', 'bert.encoder.layer.2.attention.self.query.bias', 'bert.encoder.layer.2.attention.self.key.weight', 'bert.encoder.layer.2.attention.self.key.bias', 'bert.encoder.layer.2.attention.self.value.weight', 'bert.encoder.layer.2.attention.self.value.bias', 'bert.encoder.layer.2.attention.output.dense.weight', 'bert.encoder.layer.2.attention.output.dense.bias', 'bert.encoder.layer.2.attention.output.LayerNorm.weight', 'bert.encoder.layer.2.attention.output.LayerNorm.bias', 'bert.encoder.layer.2.intermediate.dense_act.weight', 'bert.encoder.layer.2.intermediate.dense_act.bias', 'bert.encoder.layer.2.output.dense.weight', 'bert.encoder.layer.2.output.dense.bias', 'bert.encoder.layer.2.output.LayerNorm.weight', 'bert.encoder.layer.2.output.LayerNorm.bias', 'bert.encoder.layer.3.attention.self.query.weight', 'bert.encoder.layer.3.attention.self.query.bias', 'bert.encoder.layer.3.attention.self.key.weight', 'bert.encoder.layer.3.attention.self.key.bias', 'bert.encoder.layer.3.attention.self.value.weight', 'bert.encoder.layer.3.attention.self.value.bias', 'bert.encoder.layer.3.attention.output.dense.weight', 'bert.encoder.layer.3.attention.output.dense.bias', 'bert.encoder.layer.3.attention.output.LayerNorm.weight', 'bert.encoder.layer.3.attention.output.LayerNorm.bias', 'bert.encoder.layer.3.intermediate.dense_act.weight', 'bert.encoder.layer.3.intermediate.dense_act.bias', 'bert.encoder.layer.3.output.dense.weight', 'bert.encoder.layer.3.output.dense.bias', 'bert.encoder.layer.3.output.LayerNorm.weight', 'bert.encoder.layer.3.output.LayerNorm.bias', 'bert.encoder.layer.4.attention.self.query.weight', 'bert.encoder.layer.4.attention.self.query.bias', 'bert.encoder.layer.4.attention.self.key.weight', 'bert.encoder.layer.4.attention.self.key.bias', 'bert.encoder.layer.4.attention.self.value.weight', 'bert.encoder.layer.4.attention.self.value.bias', 'bert.encoder.layer.4.attention.output.dense.weight', 'bert.encoder.layer.4.attention.output.dense.bias', 'bert.encoder.layer.4.attention.output.LayerNorm.weight', 'bert.encoder.layer.4.attention.output.LayerNorm.bias', 'bert.encoder.layer.4.intermediate.dense_act.weight', 'bert.encoder.layer.4.intermediate.dense_act.bias', 'bert.encoder.layer.4.output.dense.weight', 'bert.encoder.layer.4.output.dense.bias', 'bert.encoder.layer.4.output.LayerNorm.weight', 'bert.encoder.layer.4.output.LayerNorm.bias', 'bert.encoder.layer.5.attention.self.query.weight', 'bert.encoder.layer.5.attention.self.query.bias', 'bert.encoder.layer.5.attention.self.key.weight', 'bert.encoder.layer.5.attention.self.key.bias', 'bert.encoder.layer.5.attention.self.value.weight', 'bert.encoder.layer.5.attention.self.value.bias', 'bert.encoder.layer.5.attention.output.dense.weight', 'bert.encoder.layer.5.attention.output.dense.bias', 'bert.encoder.layer.5.attention.output.LayerNorm.weight', 'bert.encoder.layer.5.attention.output.LayerNorm.bias', 'bert.encoder.layer.5.intermediate.dense_act.weight', 'bert.encoder.layer.5.intermediate.dense_act.bias', 'bert.encoder.layer.5.output.dense.weight', 'bert.encoder.layer.5.output.dense.bias', 'bert.encoder.layer.5.output.LayerNorm.weight', 'bert.encoder.layer.5.output.LayerNorm.bias', 'bert.encoder.layer.6.attention.self.query.weight', 'bert.encoder.layer.6.attention.self.query.bias', 'bert.encoder.layer.6.attention.self.key.weight', 'bert.encoder.layer.6.attention.self.key.bias', 'bert.encoder.layer.6.attention.self.value.weight', 'bert.encoder.layer.6.attention.self.value.bias', 'bert.encoder.layer.6.attention.output.dense.weight', 'bert.encoder.layer.6.attention.output.dense.bias', 'bert.encoder.layer.6.attention.output.LayerNorm.weight', 'bert.encoder.layer.6.attention.output.LayerNorm.bias', 'bert.encoder.layer.6.intermediate.dense_act.weight', 'bert.encoder.layer.6.intermediate.dense_act.bias', 'bert.encoder.layer.6.output.dense.weight', 'bert.encoder.layer.6.output.dense.bias', 'bert.encoder.layer.6.output.LayerNorm.weight', 'bert.encoder.layer.6.output.LayerNorm.bias', 'bert.encoder.layer.7.attention.self.query.weight', 'bert.encoder.layer.7.attention.self.query.bias', 'bert.encoder.layer.7.attention.self.key.weight', 'bert.encoder.layer.7.attention.self.key.bias', 'bert.encoder.layer.7.attention.self.value.weight', 'bert.encoder.layer.7.attention.self.value.bias', 'bert.encoder.layer.7.attention.output.dense.weight', 'bert.encoder.layer.7.attention.output.dense.bias', 'bert.encoder.layer.7.attention.output.LayerNorm.weight', 'bert.encoder.layer.7.attention.output.LayerNorm.bias', 'bert.encoder.layer.7.intermediate.dense_act.weight', 'bert.encoder.layer.7.intermediate.dense_act.bias', 'bert.encoder.layer.7.output.dense.weight', 'bert.encoder.layer.7.output.dense.bias', 'bert.encoder.layer.7.output.LayerNorm.weight', 'bert.encoder.layer.7.output.LayerNorm.bias', 'bert.encoder.layer.8.attention.self.query.weight', 'bert.encoder.layer.8.attention.self.query.bias', 'bert.encoder.layer.8.attention.self.key.weight', 'bert.encoder.layer.8.attention.self.key.bias', 'bert.encoder.layer.8.attention.self.value.weight', 'bert.encoder.layer.8.attention.self.value.bias', 'bert.encoder.layer.8.attention.output.dense.weight', 'bert.encoder.layer.8.attention.output.dense.bias', 'bert.encoder.layer.8.attention.output.LayerNorm.weight', 'bert.encoder.layer.8.attention.output.LayerNorm.bias', 'bert.encoder.layer.8.intermediate.dense_act.weight', 'bert.encoder.layer.8.intermediate.dense_act.bias', 'bert.encoder.layer.8.output.dense.weight', 'bert.encoder.layer.8.output.dense.bias', 'bert.encoder.layer.8.output.LayerNorm.weight', 'bert.encoder.layer.8.output.LayerNorm.bias', 'bert.encoder.layer.9.attention.self.query.weight', 'bert.encoder.layer.9.attention.self.query.bias', 'bert.encoder.layer.9.attention.self.key.weight', 'bert.encoder.layer.9.attention.self.key.bias', 'bert.encoder.layer.9.attention.self.value.weight', 'bert.encoder.layer.9.attention.self.value.bias', 'bert.encoder.layer.9.attention.output.dense.weight', 'bert.encoder.layer.9.attention.output.dense.bias', 'bert.encoder.layer.9.attention.output.LayerNorm.weight', 'bert.encoder.layer.9.attention.output.LayerNorm.bias', 'bert.encoder.layer.9.intermediate.dense_act.weight', 'bert.encoder.layer.9.intermediate.dense_act.bias', 'bert.encoder.layer.9.output.dense.weight', 'bert.encoder.layer.9.output.dense.bias', 'bert.encoder.layer.9.output.LayerNorm.weight', 'bert.encoder.layer.9.output.LayerNorm.bias', 'bert.encoder.layer.10.attention.self.query.weight', 'bert.encoder.layer.10.attention.self.query.bias', 'bert.encoder.layer.10.attention.self.key.weight', 'bert.encoder.layer.10.attention.self.key.bias', 'bert.encoder.layer.10.attention.self.value.weight', 'bert.encoder.layer.10.attention.self.value.bias', 'bert.encoder.layer.10.attention.output.dense.weight', 'bert.encoder.layer.10.attention.output.dense.bias', 'bert.encoder.layer.10.attention.output.LayerNorm.weight', 'bert.encoder.layer.10.attention.output.LayerNorm.bias', 'bert.encoder.layer.10.intermediate.dense_act.weight', 'bert.encoder.layer.10.intermediate.dense_act.bias', 'bert.encoder.layer.10.output.dense.weight', 'bert.encoder.layer.10.output.dense.bias', 'bert.encoder.layer.10.output.LayerNorm.weight', 'bert.encoder.layer.10.output.LayerNorm.bias', 'bert.encoder.layer.11.attention.self.query.weight', 'bert.encoder.layer.11.attention.self.query.bias', 'bert.encoder.layer.11.attention.self.key.weight', 'bert.encoder.layer.11.attention.self.key.bias', 'bert.encoder.layer.11.attention.self.value.weight', 'bert.encoder.layer.11.attention.self.value.bias', 'bert.encoder.layer.11.attention.output.dense.weight', 'bert.encoder.layer.11.attention.output.dense.bias', 'bert.encoder.layer.11.attention.output.LayerNorm.weight', 'bert.encoder.layer.11.attention.output.LayerNorm.bias', 'bert.encoder.layer.11.intermediate.dense_act.weight', 'bert.encoder.layer.11.intermediate.dense_act.bias', 'bert.encoder.layer.11.output.dense.weight', 'bert.encoder.layer.11.output.dense.bias', 'bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.LayerNorm.bias', 'bert.encoder.layer.12.attention.self.query.weight', 'bert.encoder.layer.12.attention.self.query.bias', 'bert.encoder.layer.12.attention.self.key.weight', 'bert.encoder.layer.12.attention.self.key.bias', 'bert.encoder.layer.12.attention.self.value.weight', 'bert.encoder.layer.12.attention.self.value.bias', 'bert.encoder.layer.12.attention.output.dense.weight', 'bert.encoder.layer.12.attention.output.dense.bias', 'bert.encoder.layer.12.attention.output.LayerNorm.weight', 'bert.encoder.layer.12.attention.output.LayerNorm.bias', 'bert.encoder.layer.12.intermediate.dense_act.weight', 'bert.encoder.layer.12.intermediate.dense_act.bias', 'bert.encoder.layer.12.output.dense.weight', 'bert.encoder.layer.12.output.dense.bias', 'bert.encoder.layer.12.output.LayerNorm.weight', 'bert.encoder.layer.12.output.LayerNorm.bias', 'bert.encoder.layer.13.attention.self.query.weight', 'bert.encoder.layer.13.attention.self.query.bias', 'bert.encoder.layer.13.attention.self.key.weight', 'bert.encoder.layer.13.attention.self.key.bias', 'bert.encoder.layer.13.attention.self.value.weight', 'bert.encoder.layer.13.attention.self.value.bias', 'bert.encoder.layer.13.attention.output.dense.weight', 'bert.encoder.layer.13.attention.output.dense.bias', 'bert.encoder.layer.13.attention.output.LayerNorm.weight', 'bert.encoder.layer.13.attention.output.LayerNorm.bias', 'bert.encoder.layer.13.intermediate.dense_act.weight', 'bert.encoder.layer.13.intermediate.dense_act.bias', 'bert.encoder.layer.13.output.dense.weight', 'bert.encoder.layer.13.output.dense.bias', 'bert.encoder.layer.13.output.LayerNorm.weight', 'bert.encoder.layer.13.output.LayerNorm.bias', 'bert.encoder.layer.14.attention.self.query.weight', 'bert.encoder.layer.14.attention.self.query.bias', 'bert.encoder.layer.14.attention.self.key.weight', 'bert.encoder.layer.14.attention.self.key.bias', 'bert.encoder.layer.14.attention.self.value.weight', 'bert.encoder.layer.14.attention.self.value.bias', 'bert.encoder.layer.14.attention.output.dense.weight', 'bert.encoder.layer.14.attention.output.dense.bias', 'bert.encoder.layer.14.attention.output.LayerNorm.weight', 'bert.encoder.layer.14.attention.output.LayerNorm.bias', 'bert.encoder.layer.14.intermediate.dense_act.weight', 'bert.encoder.layer.14.intermediate.dense_act.bias', 'bert.encoder.layer.14.output.dense.weight', 'bert.encoder.layer.14.output.dense.bias', 'bert.encoder.layer.14.output.LayerNorm.weight', 'bert.encoder.layer.14.output.LayerNorm.bias', 'bert.encoder.layer.15.attention.self.query.weight', 'bert.encoder.layer.15.attention.self.query.bias', 'bert.encoder.layer.15.attention.self.key.weight', 'bert.encoder.layer.15.attention.self.key.bias', 'bert.encoder.layer.15.attention.self.value.weight', 'bert.encoder.layer.15.attention.self.value.bias', 'bert.encoder.layer.15.attention.output.dense.weight', 'bert.encoder.layer.15.attention.output.dense.bias', 'bert.encoder.layer.15.attention.output.LayerNorm.weight', 'bert.encoder.layer.15.attention.output.LayerNorm.bias', 'bert.encoder.layer.15.intermediate.dense_act.weight', 'bert.encoder.layer.15.intermediate.dense_act.bias', 'bert.encoder.layer.15.output.dense.weight', 'bert.encoder.layer.15.output.dense.bias', 'bert.encoder.layer.15.output.LayerNorm.weight', 'bert.encoder.layer.15.output.LayerNorm.bias', 'bert.encoder.layer.16.attention.self.query.weight', 'bert.encoder.layer.16.attention.self.query.bias', 'bert.encoder.layer.16.attention.self.key.weight', 'bert.encoder.layer.16.attention.self.key.bias', 'bert.encoder.layer.16.attention.self.value.weight', 'bert.encoder.layer.16.attention.self.value.bias', 'bert.encoder.layer.16.attention.output.dense.weight', 'bert.encoder.layer.16.attention.output.dense.bias', 'bert.encoder.layer.16.attention.output.LayerNorm.weight', 'bert.encoder.layer.16.attention.output.LayerNorm.bias', 'bert.encoder.layer.16.intermediate.dense_act.weight', 'bert.encoder.layer.16.intermediate.dense_act.bias', 'bert.encoder.layer.16.output.dense.weight', 'bert.encoder.layer.16.output.dense.bias', 'bert.encoder.layer.16.output.LayerNorm.weight', 'bert.encoder.layer.16.output.LayerNorm.bias', 'bert.encoder.layer.17.attention.self.query.weight', 'bert.encoder.layer.17.attention.self.query.bias', 'bert.encoder.layer.17.attention.self.key.weight', 'bert.encoder.layer.17.attention.self.key.bias', 'bert.encoder.layer.17.attention.self.value.weight', 'bert.encoder.layer.17.attention.self.value.bias', 'bert.encoder.layer.17.attention.output.dense.weight', 'bert.encoder.layer.17.attention.output.dense.bias', 'bert.encoder.layer.17.attention.output.LayerNorm.weight', 'bert.encoder.layer.17.attention.output.LayerNorm.bias', 'bert.encoder.layer.17.intermediate.dense_act.weight', 'bert.encoder.layer.17.intermediate.dense_act.bias', 'bert.encoder.layer.17.output.dense.weight', 'bert.encoder.layer.17.output.dense.bias', 'bert.encoder.layer.17.output.LayerNorm.weight', 'bert.encoder.layer.17.output.LayerNorm.bias', 'bert.encoder.layer.18.attention.self.query.weight', 'bert.encoder.layer.18.attention.self.query.bias', 'bert.encoder.layer.18.attention.self.key.weight', 'bert.encoder.layer.18.attention.self.key.bias', 'bert.encoder.layer.18.attention.self.value.weight', 'bert.encoder.layer.18.attention.self.value.bias', 'bert.encoder.layer.18.attention.output.dense.weight', 'bert.encoder.layer.18.attention.output.dense.bias', 'bert.encoder.layer.18.attention.output.LayerNorm.weight', 'bert.encoder.layer.18.attention.output.LayerNorm.bias', 'bert.encoder.layer.18.intermediate.dense_act.weight', 'bert.encoder.layer.18.intermediate.dense_act.bias', 'bert.encoder.layer.18.output.dense.weight', 'bert.encoder.layer.18.output.dense.bias', 'bert.encoder.layer.18.output.LayerNorm.weight', 'bert.encoder.layer.18.output.LayerNorm.bias', 'bert.encoder.layer.19.attention.self.query.weight', 'bert.encoder.layer.19.attention.self.query.bias', 'bert.encoder.layer.19.attention.self.key.weight', 'bert.encoder.layer.19.attention.self.key.bias', 'bert.encoder.layer.19.attention.self.value.weight', 'bert.encoder.layer.19.attention.self.value.bias', 'bert.encoder.layer.19.attention.output.dense.weight', 'bert.encoder.layer.19.attention.output.dense.bias', 'bert.encoder.layer.19.attention.output.LayerNorm.weight', 'bert.encoder.layer.19.attention.output.LayerNorm.bias', 'bert.encoder.layer.19.intermediate.dense_act.weight', 'bert.encoder.layer.19.intermediate.dense_act.bias', 'bert.encoder.layer.19.output.dense.weight', 'bert.encoder.layer.19.output.dense.bias', 'bert.encoder.layer.19.output.LayerNorm.weight', 'bert.encoder.layer.19.output.LayerNorm.bias', 'bert.encoder.layer.20.attention.self.query.weight', 'bert.encoder.layer.20.attention.self.query.bias', 'bert.encoder.layer.20.attention.self.key.weight', 'bert.encoder.layer.20.attention.self.key.bias', 'bert.encoder.layer.20.attention.self.value.weight', 'bert.encoder.layer.20.attention.self.value.bias', 'bert.encoder.layer.20.attention.output.dense.weight', 'bert.encoder.layer.20.attention.output.dense.bias', 'bert.encoder.layer.20.attention.output.LayerNorm.weight', 'bert.encoder.layer.20.attention.output.LayerNorm.bias', 'bert.encoder.layer.20.intermediate.dense_act.weight', 'bert.encoder.layer.20.intermediate.dense_act.bias', 'bert.encoder.layer.20.output.dense.weight', 'bert.encoder.layer.20.output.dense.bias', 'bert.encoder.layer.20.output.LayerNorm.weight', 'bert.encoder.layer.20.output.LayerNorm.bias', 'bert.encoder.layer.21.attention.self.query.weight', 'bert.encoder.layer.21.attention.self.query.bias', 'bert.encoder.layer.21.attention.self.key.weight', 'bert.encoder.layer.21.attention.self.key.bias', 'bert.encoder.layer.21.attention.self.value.weight', 'bert.encoder.layer.21.attention.self.value.bias', 'bert.encoder.layer.21.attention.output.dense.weight', 'bert.encoder.layer.21.attention.output.dense.bias', 'bert.encoder.layer.21.attention.output.LayerNorm.weight', 'bert.encoder.layer.21.attention.output.LayerNorm.bias', 'bert.encoder.layer.21.intermediate.dense_act.weight', 'bert.encoder.layer.21.intermediate.dense_act.bias', 'bert.encoder.layer.21.output.dense.weight', 'bert.encoder.layer.21.output.dense.bias', 'bert.encoder.layer.21.output.LayerNorm.weight', 'bert.encoder.layer.21.output.LayerNorm.bias', 'bert.encoder.layer.22.attention.self.query.weight', 'bert.encoder.layer.22.attention.self.query.bias', 'bert.encoder.layer.22.attention.self.key.weight', 'bert.encoder.layer.22.attention.self.key.bias', 'bert.encoder.layer.22.attention.self.value.weight', 'bert.encoder.layer.22.attention.self.value.bias', 'bert.encoder.layer.22.attention.output.dense.weight', 'bert.encoder.layer.22.attention.output.dense.bias', 'bert.encoder.layer.22.attention.output.LayerNorm.weight', 'bert.encoder.layer.22.attention.output.LayerNorm.bias', 'bert.encoder.layer.22.intermediate.dense_act.weight', 'bert.encoder.layer.22.intermediate.dense_act.bias', 'bert.encoder.layer.22.output.dense.weight', 'bert.encoder.layer.22.output.dense.bias', 'bert.encoder.layer.22.output.LayerNorm.weight', 'bert.encoder.layer.22.output.LayerNorm.bias', 'bert.encoder.layer.23.attention.self.query.weight', 'bert.encoder.layer.23.attention.self.query.bias', 'bert.encoder.layer.23.attention.self.key.weight', 'bert.encoder.layer.23.attention.self.key.bias', 'bert.encoder.layer.23.attention.self.value.weight', 'bert.encoder.layer.23.attention.self.value.bias', 'bert.encoder.layer.23.attention.output.dense.weight', 'bert.encoder.layer.23.attention.output.dense.bias', 'bert.encoder.layer.23.attention.output.LayerNorm.weight', 'bert.encoder.layer.23.attention.output.LayerNorm.bias', 'bert.encoder.layer.23.intermediate.dense_act.weight', 'bert.encoder.layer.23.intermediate.dense_act.bias', 'bert.encoder.layer.23.output.dense.weight', 'bert.encoder.layer.23.output.dense.bias', 'bert.encoder.layer.23.output.LayerNorm.weight', 'bert.encoder.layer.23.output.LayerNorm.bias', 'bert.pooler.dense_act.weight', 'bert.pooler.dense_act.bias', 'cls.predictions.bias', 'cls.predictions.transform.dense_act.weight', 'cls.predictions.transform.dense_act.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias']
    optimizer_state_keys = [(0, 'exp_avg'), (0, 'exp_avg_sq'), (1, 'exp_avg'), (1, 'exp_avg_sq'), (2, 'exp_avg'), (2, 'exp_avg_sq'), (3, 'exp_avg'), (3, 'exp_avg_sq'), (4, 'exp_avg'), (4, 'exp_avg_sq'), (5, 'exp_avg'), (5, 'exp_avg_sq'), (6, 'exp_avg'), (6, 'exp_avg_sq'), (7, 'exp_avg'), (7, 'exp_avg_sq'), (8, 'exp_avg'), (8, 'exp_avg_sq'), (9, 'exp_avg'), (9, 'exp_avg_sq'), (10, 'exp_avg'), (10, 'exp_avg_sq'), (11, 'exp_avg'), (11, 'exp_avg_sq'), (12, 'exp_avg'), (12, 'exp_avg_sq'), (13, 'exp_avg'), (13, 'exp_avg_sq'), (14, 'exp_avg'), (14, 'exp_avg_sq'), (15, 'exp_avg'), (15, 'exp_avg_sq'), (16, 'exp_avg'), (16, 'exp_avg_sq'), (17, 'exp_avg'), (17, 'exp_avg_sq'), (18, 'exp_avg'), (18, 'exp_avg_sq'), (19, 'exp_avg'), (19, 'exp_avg_sq'), (20, 'exp_avg'), (20, 'exp_avg_sq'), (21, 'exp_avg'), (21, 'exp_avg_sq'), (22, 'exp_avg'), (22, 'exp_avg_sq'), (23, 'exp_avg'), (23, 'exp_avg_sq'), (24, 'exp_avg'), (24, 'exp_avg_sq'), (25, 'exp_avg'), (25, 'exp_avg_sq'), (26, 'exp_avg'), (26, 'exp_avg_sq'), (27, 'exp_avg'), (27, 'exp_avg_sq'), (28, 'exp_avg'), (28, 'exp_avg_sq'), (29, 'exp_avg'), (29, 'exp_avg_sq'), (30, 'exp_avg'), (30, 'exp_avg_sq'), (31, 'exp_avg'), (31, 'exp_avg_sq'), (32, 'exp_avg'), (32, 'exp_avg_sq'), (33, 'exp_avg'), (33, 'exp_avg_sq'), (34, 'exp_avg'), (34, 'exp_avg_sq'), (35, 'exp_avg'), (35, 'exp_avg_sq'), (36, 'exp_avg'), (36, 'exp_avg_sq'), (37, 'exp_avg'), (37, 'exp_avg_sq'), (38, 'exp_avg'), (38, 'exp_avg_sq'), (39, 'exp_avg'), (39, 'exp_avg_sq'), (40, 'exp_avg'), (40, 'exp_avg_sq'), (41, 'exp_avg'), (41, 'exp_avg_sq'), (42, 'exp_avg'), (42, 'exp_avg_sq'), (43, 'exp_avg'), (43, 'exp_avg_sq'), (44, 'exp_avg'), (44, 'exp_avg_sq'), (45, 'exp_avg'), (45, 'exp_avg_sq'), (46, 'exp_avg'), (46, 'exp_avg_sq'), (47, 'exp_avg'), (47, 'exp_avg_sq'), (48, 'exp_avg'), (48, 'exp_avg_sq'), (49, 'exp_avg'), (49, 'exp_avg_sq'), (50, 'exp_avg'), (50, 'exp_avg_sq'), (51, 'exp_avg'), (51, 'exp_avg_sq'), (52, 'exp_avg'), (52, 'exp_avg_sq'), (53, 'exp_avg'), (53, 'exp_avg_sq'), (54, 'exp_avg'), (54, 'exp_avg_sq'), (55, 'exp_avg'), (55, 'exp_avg_sq'), (56, 'exp_avg'), (56, 'exp_avg_sq'), (57, 'exp_avg'), (57, 'exp_avg_sq'), (58, 'exp_avg'), (58, 'exp_avg_sq'), (59, 'exp_avg'), (59, 'exp_avg_sq'), (60, 'exp_avg'), (60, 'exp_avg_sq'), (61, 'exp_avg'), (61, 'exp_avg_sq'), (62, 'exp_avg'), (62, 'exp_avg_sq'), (63, 'exp_avg'), (63, 'exp_avg_sq'), (64, 'exp_avg'), (64, 'exp_avg_sq'), (65, 'exp_avg'), (65, 'exp_avg_sq'), (66, 'exp_avg'), (66, 'exp_avg_sq'), (67, 'exp_avg'), (67, 'exp_avg_sq'), (68, 'exp_avg'), (68, 'exp_avg_sq'), (69, 'exp_avg'), (69, 'exp_avg_sq'), (70, 'exp_avg'), (70, 'exp_avg_sq'), (71, 'exp_avg'), (71, 'exp_avg_sq'), (72, 'exp_avg'), (72, 'exp_avg_sq'), (73, 'exp_avg'), (73, 'exp_avg_sq'), (74, 'exp_avg'), (74, 'exp_avg_sq'), (75, 'exp_avg'), (75, 'exp_avg_sq'), (76, 'exp_avg'), (76, 'exp_avg_sq'), (77, 'exp_avg'), (77, 'exp_avg_sq'), (78, 'exp_avg'), (78, 'exp_avg_sq'), (79, 'exp_avg'), (79, 'exp_avg_sq'), (80, 'exp_avg'), (80, 'exp_avg_sq'), (81, 'exp_avg'), (81, 'exp_avg_sq'), (82, 'exp_avg'), (82, 'exp_avg_sq'), (83, 'exp_avg'), (83, 'exp_avg_sq'), (84, 'exp_avg'), (84, 'exp_avg_sq'), (85, 'exp_avg'), (85, 'exp_avg_sq'), (86, 'exp_avg'), (86, 'exp_avg_sq'), (87, 'exp_avg'), (87, 'exp_avg_sq'), (88, 'exp_avg'), (88, 'exp_avg_sq'), (89, 'exp_avg'), (89, 'exp_avg_sq'), (90, 'exp_avg'), (90, 'exp_avg_sq'), (91, 'exp_avg'), (91, 'exp_avg_sq'), (92, 'exp_avg'), (92, 'exp_avg_sq'), (93, 'exp_avg'), (93, 'exp_avg_sq'), (94, 'exp_avg'), (94, 'exp_avg_sq'), (95, 'exp_avg'), (95, 'exp_avg_sq'), (96, 'exp_avg'), (96, 'exp_avg_sq'), (97, 'exp_avg'), (97, 'exp_avg_sq'), (98, 'exp_avg'), (98, 'exp_avg_sq'), (99, 'exp_avg'), (99, 'exp_avg_sq'), (100, 'exp_avg'), (100, 'exp_avg_sq'), (101, 'exp_avg'), (101, 'exp_avg_sq'), (102, 'exp_avg'), (102, 'exp_avg_sq'), (103, 'exp_avg'), (103, 'exp_avg_sq'), (104, 'exp_avg'), (104, 'exp_avg_sq'), (105, 'exp_avg'), (105, 'exp_avg_sq'), (106, 'exp_avg'), (106, 'exp_avg_sq'), (107, 'exp_avg'), (107, 'exp_avg_sq'), (108, 'exp_avg'), (108, 'exp_avg_sq'), (109, 'exp_avg'), (109, 'exp_avg_sq'), (110, 'exp_avg'), (110, 'exp_avg_sq'), (111, 'exp_avg'), (111, 'exp_avg_sq'), (112, 'exp_avg'), (112, 'exp_avg_sq'), (113, 'exp_avg'), (113, 'exp_avg_sq'), (114, 'exp_avg'), (114, 'exp_avg_sq'), (115, 'exp_avg'), (115, 'exp_avg_sq'), (116, 'exp_avg'), (116, 'exp_avg_sq'), (117, 'exp_avg'), (117, 'exp_avg_sq'), (118, 'exp_avg'), (118, 'exp_avg_sq'), (119, 'exp_avg'), (119, 'exp_avg_sq'), (120, 'exp_avg'), (120, 'exp_avg_sq'), (121, 'exp_avg'), (121, 'exp_avg_sq'), (122, 'exp_avg'), (122, 'exp_avg_sq'), (123, 'exp_avg'), (123, 'exp_avg_sq'), (124, 'exp_avg'), (124, 'exp_avg_sq'), (125, 'exp_avg'), (125, 'exp_avg_sq'), (126, 'exp_avg'), (126, 'exp_avg_sq'), (127, 'exp_avg'), (127, 'exp_avg_sq'), (128, 'exp_avg'), (128, 'exp_avg_sq'), (129, 'exp_avg'), (129, 'exp_avg_sq'), (130, 'exp_avg'), (130, 'exp_avg_sq'), (131, 'exp_avg'), (131, 'exp_avg_sq'), (132, 'exp_avg'), (132, 'exp_avg_sq'), (133, 'exp_avg'), (133, 'exp_avg_sq'), (134, 'exp_avg'), (134, 'exp_avg_sq'), (135, 'exp_avg'), (135, 'exp_avg_sq'), (136, 'exp_avg'), (136, 'exp_avg_sq'), (137, 'exp_avg'), (137, 'exp_avg_sq'), (138, 'exp_avg'), (138, 'exp_avg_sq'), (139, 'exp_avg'), (139, 'exp_avg_sq'), (140, 'exp_avg'), (140, 'exp_avg_sq'), (141, 'exp_avg'), (141, 'exp_avg_sq'), (142, 'exp_avg'), (142, 'exp_avg_sq'), (143, 'exp_avg'), (143, 'exp_avg_sq'), (144, 'exp_avg'), (144, 'exp_avg_sq'), (145, 'exp_avg'), (145, 'exp_avg_sq'), (146, 'exp_avg'), (146, 'exp_avg_sq'), (147, 'exp_avg'), (147, 'exp_avg_sq'), (148, 'exp_avg'), (148, 'exp_avg_sq'), (149, 'exp_avg'), (149, 'exp_avg_sq'), (150, 'exp_avg'), (150, 'exp_avg_sq'), (151, 'exp_avg'), (151, 'exp_avg_sq'), (152, 'exp_avg'), (152, 'exp_avg_sq'), (153, 'exp_avg'), (153, 'exp_avg_sq'), (154, 'exp_avg'), (154, 'exp_avg_sq'), (155, 'exp_avg'), (155, 'exp_avg_sq'), (156, 'exp_avg'), (156, 'exp_avg_sq'), (157, 'exp_avg'), (157, 'exp_avg_sq'), (158, 'exp_avg'), (158, 'exp_avg_sq'), (159, 'exp_avg'), (159, 'exp_avg_sq'), (160, 'exp_avg'), (160, 'exp_avg_sq'), (161, 'exp_avg'), (161, 'exp_avg_sq'), (162, 'exp_avg'), (162, 'exp_avg_sq'), (163, 'exp_avg'), (163, 'exp_avg_sq'), (164, 'exp_avg'), (164, 'exp_avg_sq'), (165, 'exp_avg'), (165, 'exp_avg_sq'), (166, 'exp_avg'), (166, 'exp_avg_sq'), (167, 'exp_avg'), (167, 'exp_avg_sq'), (168, 'exp_avg'), (168, 'exp_avg_sq'), (169, 'exp_avg'), (169, 'exp_avg_sq'), (170, 'exp_avg'), (170, 'exp_avg_sq'), (171, 'exp_avg'), (171, 'exp_avg_sq'), (172, 'exp_avg'), (172, 'exp_avg_sq'), (173, 'exp_avg'), (173, 'exp_avg_sq'), (174, 'exp_avg'), (174, 'exp_avg_sq'), (175, 'exp_avg'), (175, 'exp_avg_sq'), (176, 'exp_avg'), (176, 'exp_avg_sq'), (177, 'exp_avg'), (177, 'exp_avg_sq'), (178, 'exp_avg'), (178, 'exp_avg_sq'), (179, 'exp_avg'), (179, 'exp_avg_sq'), (180, 'exp_avg'), (180, 'exp_avg_sq'), (181, 'exp_avg'), (181, 'exp_avg_sq'), (182, 'exp_avg'), (182, 'exp_avg_sq'), (183, 'exp_avg'), (183, 'exp_avg_sq'), (184, 'exp_avg'), (184, 'exp_avg_sq'), (185, 'exp_avg'), (185, 'exp_avg_sq'), (186, 'exp_avg'), (186, 'exp_avg_sq'), (187, 'exp_avg'), (187, 'exp_avg_sq'), (188, 'exp_avg'), (188, 'exp_avg_sq'), (189, 'exp_avg'), (189, 'exp_avg_sq'), (190, 'exp_avg'), (190, 'exp_avg_sq'), (191, 'exp_avg'), (191, 'exp_avg_sq'), (192, 'exp_avg'), (192, 'exp_avg_sq'), (193, 'exp_avg'), (193, 'exp_avg_sq'), (194, 'exp_avg'), (194, 'exp_avg_sq'), (195, 'exp_avg'), (195, 'exp_avg_sq'), (196, 'exp_avg'), (196, 'exp_avg_sq'), (197, 'exp_avg'), (197, 'exp_avg_sq'), (198, 'exp_avg'), (198, 'exp_avg_sq'), (199, 'exp_avg'), (199, 'exp_avg_sq'), (200, 'exp_avg'), (200, 'exp_avg_sq'), (201, 'exp_avg'), (201, 'exp_avg_sq'), (202, 'exp_avg'), (202, 'exp_avg_sq'), (203, 'exp_avg'), (203, 'exp_avg_sq'), (204, 'exp_avg'), (204, 'exp_avg_sq'), (205, 'exp_avg'), (205, 'exp_avg_sq'), (206, 'exp_avg'), (206, 'exp_avg_sq'), (207, 'exp_avg'), (207, 'exp_avg_sq'), (208, 'exp_avg'), (208, 'exp_avg_sq'), (209, 'exp_avg'), (209, 'exp_avg_sq'), (210, 'exp_avg'), (210, 'exp_avg_sq'), (211, 'exp_avg'), (211, 'exp_avg_sq'), (212, 'exp_avg'), (212, 'exp_avg_sq'), (213, 'exp_avg'), (213, 'exp_avg_sq'), (214, 'exp_avg'), (214, 'exp_avg_sq'), (215, 'exp_avg'), (215, 'exp_avg_sq'), (216, 'exp_avg'), (216, 'exp_avg_sq'), (217, 'exp_avg'), (217, 'exp_avg_sq'), (218, 'exp_avg'), (218, 'exp_avg_sq'), (219, 'exp_avg'), (219, 'exp_avg_sq'), (220, 'exp_avg'), (220, 'exp_avg_sq'), (221, 'exp_avg'), (221, 'exp_avg_sq'), (222, 'exp_avg'), (222, 'exp_avg_sq'), (223, 'exp_avg'), (223, 'exp_avg_sq'), (224, 'exp_avg'), (224, 'exp_avg_sq'), (225, 'exp_avg'), (225, 'exp_avg_sq'), (226, 'exp_avg'), (226, 'exp_avg_sq'), (227, 'exp_avg'), (227, 'exp_avg_sq'), (228, 'exp_avg'), (228, 'exp_avg_sq'), (229, 'exp_avg'), (229, 'exp_avg_sq'), (230, 'exp_avg'), (230, 'exp_avg_sq'), (231, 'exp_avg'), (231, 'exp_avg_sq'), (232, 'exp_avg'), (232, 'exp_avg_sq'), (233, 'exp_avg'), (233, 'exp_avg_sq'), (234, 'exp_avg'), (234, 'exp_avg_sq'), (235, 'exp_avg'), (235, 'exp_avg_sq'), (236, 'exp_avg'), (236, 'exp_avg_sq'), (237, 'exp_avg'), (237, 'exp_avg_sq'), (238, 'exp_avg'), (238, 'exp_avg_sq'), (239, 'exp_avg'), (239, 'exp_avg_sq'), (240, 'exp_avg'), (240, 'exp_avg_sq'), (241, 'exp_avg'), (241, 'exp_avg_sq'), (242, 'exp_avg'), (242, 'exp_avg_sq'), (243, 'exp_avg'), (243, 'exp_avg_sq'), (244, 'exp_avg'), (244, 'exp_avg_sq'), (245, 'exp_avg'), (245, 'exp_avg_sq'), (246, 'exp_avg'), (246, 'exp_avg_sq'), (247, 'exp_avg'), (247, 'exp_avg_sq'), (248, 'exp_avg'), (248, 'exp_avg_sq'), (249, 'exp_avg'), (249, 'exp_avg_sq'), (250, 'exp_avg'), (250, 'exp_avg_sq'), (251, 'exp_avg'), (251, 'exp_avg_sq'), (252, 'exp_avg'), (252, 'exp_avg_sq'), (253, 'exp_avg'), (253, 'exp_avg_sq'), (254, 'exp_avg'), (254, 'exp_avg_sq'), (255, 'exp_avg'), (255, 'exp_avg_sq'), (256, 'exp_avg'), (256, 'exp_avg_sq'), (257, 'exp_avg'), (257, 'exp_avg_sq'), (258, 'exp_avg'), (258, 'exp_avg_sq'), (259, 'exp_avg'), (259, 'exp_avg_sq'), (260, 'exp_avg'), (260, 'exp_avg_sq'), (261, 'exp_avg'), (261, 'exp_avg_sq'), (262, 'exp_avg'), (262, 'exp_avg_sq'), (263, 'exp_avg'), (263, 'exp_avg_sq'), (264, 'exp_avg'), (264, 'exp_avg_sq'), (265, 'exp_avg'), (265, 'exp_avg_sq'), (266, 'exp_avg'), (266, 'exp_avg_sq'), (267, 'exp_avg'), (267, 'exp_avg_sq'), (268, 'exp_avg'), (268, 'exp_avg_sq'), (269, 'exp_avg'), (269, 'exp_avg_sq'), (270, 'exp_avg'), (270, 'exp_avg_sq'), (271, 'exp_avg'), (271, 'exp_avg_sq'), (272, 'exp_avg'), (272, 'exp_avg_sq'), (273, 'exp_avg'), (273, 'exp_avg_sq'), (274, 'exp_avg'), (274, 'exp_avg_sq'), (275, 'exp_avg'), (275, 'exp_avg_sq'), (276, 'exp_avg'), (276, 'exp_avg_sq'), (277, 'exp_avg'), (277, 'exp_avg_sq'), (278, 'exp_avg'), (278, 'exp_avg_sq'), (279, 'exp_avg'), (279, 'exp_avg_sq'), (280, 'exp_avg'), (280, 'exp_avg_sq'), (281, 'exp_avg'), (281, 'exp_avg_sq'), (282, 'exp_avg'), (282, 'exp_avg_sq'), (283, 'exp_avg'), (283, 'exp_avg_sq'), (284, 'exp_avg'), (284, 'exp_avg_sq'), (285, 'exp_avg'), (285, 'exp_avg_sq'), (286, 'exp_avg'), (286, 'exp_avg_sq'), (287, 'exp_avg'), (287, 'exp_avg_sq'), (288, 'exp_avg'), (288, 'exp_avg_sq'), (289, 'exp_avg'), (289, 'exp_avg_sq'), (290, 'exp_avg'), (290, 'exp_avg_sq'), (291, 'exp_avg'), (291, 'exp_avg_sq'), (292, 'exp_avg'), (292, 'exp_avg_sq'), (293, 'exp_avg'), (293, 'exp_avg_sq'), (294, 'exp_avg'), (294, 'exp_avg_sq'), (295, 'exp_avg'), (295, 'exp_avg_sq'), (296, 'exp_avg'), (296, 'exp_avg_sq'), (297, 'exp_avg'), (297, 'exp_avg_sq'), (298, 'exp_avg'), (298, 'exp_avg_sq'), (299, 'exp_avg'), (299, 'exp_avg_sq'), (300, 'exp_avg'), (300, 'exp_avg_sq'), (301, 'exp_avg'), (301, 'exp_avg_sq'), (302, 'exp_avg'), (302, 'exp_avg_sq'), (303, 'exp_avg'), (303, 'exp_avg_sq'), (304, 'exp_avg'), (304, 'exp_avg_sq'), (305, 'exp_avg'), (305, 'exp_avg_sq'), (306, 'exp_avg'), (306, 'exp_avg_sq'), (307, 'exp_avg'), (307, 'exp_avg_sq'), (308, 'exp_avg'), (308, 'exp_avg_sq'), (309, 'exp_avg'), (309, 'exp_avg_sq'), (310, 'exp_avg'), (310, 'exp_avg_sq'), (311, 'exp_avg'), (311, 'exp_avg_sq'), (312, 'exp_avg'), (312, 'exp_avg_sq'), (313, 'exp_avg'), (313, 'exp_avg_sq'), (314, 'exp_avg'), (314, 'exp_avg_sq'), (315, 'exp_avg'), (315, 'exp_avg_sq'), (316, 'exp_avg'), (316, 'exp_avg_sq'), (317, 'exp_avg'), (317, 'exp_avg_sq'), (318, 'exp_avg'), (318, 'exp_avg_sq'), (319, 'exp_avg'), (319, 'exp_avg_sq'), (320, 'exp_avg'), (320, 'exp_avg_sq'), (321, 'exp_avg'), (321, 'exp_avg_sq'), (322, 'exp_avg'), (322, 'exp_avg_sq'), (323, 'exp_avg'), (323, 'exp_avg_sq'), (324, 'exp_avg'), (324, 'exp_avg_sq'), (325, 'exp_avg'), (325, 'exp_avg_sq'), (326, 'exp_avg'), (326, 'exp_avg_sq'), (327, 'exp_avg'), (327, 'exp_avg_sq'), (328, 'exp_avg'), (328, 'exp_avg_sq'), (329, 'exp_avg'), (329, 'exp_avg_sq'), (330, 'exp_avg'), (330, 'exp_avg_sq'), (331, 'exp_avg'), (331, 'exp_avg_sq'), (332, 'exp_avg'), (332, 'exp_avg_sq'), (333, 'exp_avg'), (333, 'exp_avg_sq'), (334, 'exp_avg'), (334, 'exp_avg_sq'), (335, 'exp_avg'), (335, 'exp_avg_sq'), (336, 'exp_avg'), (336, 'exp_avg_sq'), (337, 'exp_avg'), (337, 'exp_avg_sq'), (338, 'exp_avg'), (338, 'exp_avg_sq'), (339, 'exp_avg'), (339, 'exp_avg_sq'), (340, 'exp_avg'), (340, 'exp_avg_sq'), (341, 'exp_avg'), (341, 'exp_avg_sq'), (342, 'exp_avg'), (342, 'exp_avg_sq'), (343, 'exp_avg'), (343, 'exp_avg_sq'), (344, 'exp_avg'), (344, 'exp_avg_sq'), (345, 'exp_avg'), (345, 'exp_avg_sq'), (346, 'exp_avg'), (346, 'exp_avg_sq'), (347, 'exp_avg'), (347, 'exp_avg_sq'), (348, 'exp_avg'), (348, 'exp_avg_sq'), (349, 'exp_avg'), (349, 'exp_avg_sq'), (350, 'exp_avg'), (350, 'exp_avg_sq'), (351, 'exp_avg'), (351, 'exp_avg_sq'), (352, 'exp_avg'), (352, 'exp_avg_sq'), (353, 'exp_avg'), (353, 'exp_avg_sq'), (354, 'exp_avg'), (354, 'exp_avg_sq'), (355, 'exp_avg'), (355, 'exp_avg_sq'), (356, 'exp_avg'), (356, 'exp_avg_sq'), (357, 'exp_avg'), (357, 'exp_avg_sq'), (358, 'exp_avg'), (358, 'exp_avg_sq'), (359, 'exp_avg'), (359, 'exp_avg_sq'), (360, 'exp_avg'), (360, 'exp_avg_sq'), (361, 'exp_avg'), (361, 'exp_avg_sq'), (362, 'exp_avg'), (362, 'exp_avg_sq'), (363, 'exp_avg'), (363, 'exp_avg_sq'), (364, 'exp_avg'), (364, 'exp_avg_sq'), (365, 'exp_avg'), (365, 'exp_avg_sq'), (366, 'exp_avg'), (366, 'exp_avg_sq'), (367, 'exp_avg'), (367, 'exp_avg_sq'), (368, 'exp_avg'), (368, 'exp_avg_sq'), (369, 'exp_avg'), (369, 'exp_avg_sq'), (370, 'exp_avg'), (370, 'exp_avg_sq'), (371, 'exp_avg'), (371, 'exp_avg_sq'), (372, 'exp_avg'), (372, 'exp_avg_sq'), (373, 'exp_avg'), (373, 'exp_avg_sq'), (374, 'exp_avg'), (374, 'exp_avg_sq'), (375, 'exp_avg'), (375, 'exp_avg_sq'), (376, 'exp_avg'), (376, 'exp_avg_sq'), (377, 'exp_avg'), (377, 'exp_avg_sq'), (378, 'exp_avg'), (378, 'exp_avg_sq'), (379, 'exp_avg'), (379, 'exp_avg_sq'), (380, 'exp_avg'), (380, 'exp_avg_sq'), (381, 'exp_avg'), (381, 'exp_avg_sq'), (382, 'exp_avg'), (382, 'exp_avg_sq'), (383, 'exp_avg'), (383, 'exp_avg_sq'), (384, 'exp_avg'), (384, 'exp_avg_sq'), (385, 'exp_avg'), (385, 'exp_avg_sq'), (386, 'exp_avg'), (386, 'exp_avg_sq'), (387, 'exp_avg'), (387, 'exp_avg_sq'), (388, 'exp_avg'), (388, 'exp_avg_sq'), (389, 'exp_avg'), (389, 'exp_avg_sq'), (390, 'exp_avg'), (390, 'exp_avg_sq'), (391, 'exp_avg'), (391, 'exp_avg_sq'), (392, 'exp_avg'), (392, 'exp_avg_sq'), (393, 'exp_avg'), (393, 'exp_avg_sq'), (394, 'exp_avg'), (394, 'exp_avg_sq'), (395, 'exp_avg'), (395, 'exp_avg_sq'), (396, 'exp_avg'), (396, 'exp_avg_sq'), (397, 'exp_avg'), (397, 'exp_avg_sq')]

    # Prepare optimizer
    model, optimizer, grad_scaler, lr_scheduler, checkpoint, global_resume_step, criterion, epoch = prepare_model_and_optimizer(args, device, buffer_states, model_state_keys, optimizer_state_keys, sequence_output_is_dense=not args.no_dense_sequence_output)
    # Prepare the data loader.
    if is_main_process():
        tic = time.perf_counter()
    train_dataloader = lddl.torch.get_bert_pretrain_data_loader(
        args.input_dir,
        local_rank=max(args.local_rank, 0),
        vocab_file=args.vocab_file,
        data_loader_kwargs={
            'batch_size': args.train_batch_size * args.n_gpu,
            'num_workers': args.num_workers,
            'pin_memory': True,
        },
        base_seed=args.seed,
        log_dir=None if args.output_dir is None else os.path.join(args.output_dir, 'lddl_log'),
        log_level=logging.WARNING,
        start_epoch=epoch,
    )
    if is_main_process():
        print('get_bert_pretrain_data_loader took {} s!'.format(time.perf_counter() - tic))

    if is_main_process():
        dllogger.log(step="PARAMETER", data={"SEED": args.seed})
        dllogger.log(step="PARAMETER", data={"train_start": True})
        dllogger.log(step="PARAMETER", data={"batch_size_per_gpu": args.train_batch_size})
        dllogger.log(step="PARAMETER", data={"learning_rate": args.learning_rate})

    model.train()
    most_recent_ckpts_paths = []

    stats = SyncFreeStats()
    # Host Only Stats
    stats.add_stat('model_step')
    # Device/Host Sync-ed Stats
    stats.add_stat('optimizer_step', dtype=torch.int32, device_func=(lambda: optimizer.param_groups[0]['step']))
    stats.add_stat('average_loss', dtype=torch.float32, device_tensor=torch.zeros(1, dtype=torch.float32, device=device))
    stats.add_stat('learning_rate', dtype=torch.float32, device_func=(lambda: optimizer.param_groups[0]['lr']))
    if grad_scaler.is_enabled():
        # This stat only indicates a skipped step occurred.  It does not accumulate the number of skipped steps
        stats.add_stat('skip_optimizer_step', dtype=torch.float32, device_func=(lambda: grad_scaler._found_inf_per_device(optimizer)[device]))
        stats.add_stat('skipped_optimizer_steps', dtype=torch.float32, device_tensor=torch.zeros(1, dtype=torch.float32, device=device),
                                                  device_func=(lambda x: x.add_(grad_scaler._found_inf_per_device(optimizer)[device])))
    else:
        stats.add_stat('skip_optimizer_step', dtype=torch.float32)
        stats.add_stat('skipped_optimizer_steps', dtype=torch.float32)

    static_gpu_batch = None
    full_cudagraph = None
    grad_accum_cudagraph = None
    if args.cuda_graphs:
        static_gpu_batch = {
            'input_ids': torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            'token_type_ids': torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            'attention_mask': torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            'labels': torch.ones(args.train_batch_size, args.max_seq_length, dtype=torch.int64, device=device),
            'next_sentence_labels': torch.ones(args.train_batch_size, dtype=torch.int64, device=device),
        }

        side_stream = torch.cuda.Stream()

        # Warmup Steps - includes jitting fusions
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(11):
                take_training_step(args, grad_scaler, model, criterion, static_gpu_batch, stats)
                take_optimizer_step(args, lr_scheduler, optimizer, grad_scaler, device, stats)
        torch.cuda.current_stream().wait_stream(side_stream)

        # Capture Graph
        full_cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(full_cudagraph):
            take_training_step(args, grad_scaler, model, criterion, static_gpu_batch, stats)
            take_optimizer_step(args, lr_scheduler, optimizer, grad_scaler, device, stats)

        # Warmup Steps - includes jitting fusions
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                with model.no_sync():
                    take_training_step(args, grad_scaler, model, criterion, static_gpu_batch, stats)
        torch.cuda.current_stream().wait_stream(side_stream)

        # Capture Graph
        grad_accum_cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(grad_accum_cudagraph):
            with model.no_sync():
                take_training_step(args, grad_scaler, model, criterion, static_gpu_batch, stats)

    train_iter = tqdm(
        train_dataloader,
        desc="Iteration",
        disable=args.disable_progress_bar,
        total=len(train_dataloader),
    ) if is_main_process() else train_dataloader


    raw_train_start = None

    # avoid nvfuser compilation times in measuring perf with phase2 binning
    # ideally skip > 3 * num_bins fwd+bwd iterations to start measuring perf 
    skip_fwd_bwd_for_perf = 4
    if args.phase2: #we use 8 bins with phase2
        skip_fwd_bwd_for_perf = 50
    start_global = None
    create_cpu_buffers = True
    while True:
        for step, batch in enumerate(train_iter):
            if os.path.isfile("error.txt"):
                write_cpu_checkpoint_to_disk(args, buffer_states, grad_scaler, model_state_keys, optimizer_state_keys)

            # The first training step is 1 and not 0 when gradient accumulating
            # in order to avoid an optimizer step on the very first step
            stats.host_stat('model_step').add_(1)
            grad_accumulation_step = (stats.host_stat_value('model_step') % args.gradient_accumulation_steps) != 0

            if raw_train_start is None and step == skip_fwd_bwd_for_perf:
                raw_train_start = time.time()

            # Execute Model Step
            if args.cuda_graphs:
                for k in batch.keys():
                    static_gpu_batch[k].copy_(batch[k], non_blocking=True)
                if grad_accumulation_step:
                    grad_accum_cudagraph.replay()
                else:
                    full_cudagraph.replay()
            else:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                if args.allreduce_post_accumulation and grad_accumulation_step:
                    take_training_step(args, grad_scaler, model, criterion, batch, stats)
                else:
                    take_training_step(args, grad_scaler, model, criterion, batch, stats)

                take_optimizer_step(args, lr_scheduler, optimizer, grad_scaler, device, stats)

            # Log Optimizer Step
            if (not grad_accumulation_step) or timeout_sent:
                static_optimizer_step = stats.host_stat_value('model_step') // args.gradient_accumulation_steps
                dynamic_optimizer_step = static_optimizer_step - int(stats.host_stat_value('skipped_optimizer_steps')) + global_resume_step
                no_log_steps = static_optimizer_step % args.log_freq

                if create_cpu_buffers:
                    create_cpu_buffers_for_checkpointing(args, epoch, dynamic_optimizer_step, model, optimizer, grad_scaler, most_recent_ckpts_paths, buffer_states, model_state_keys, optimizer_state_keys)
                    create_cpu_buffers = False
                
                checkpoint_only_to_cpu_step(args, epoch, dynamic_optimizer_step, model, optimizer, grad_scaler, most_recent_ckpts_paths, buffer_states, model_state_keys, optimizer_state_keys)

                # Log Final Step (MAYBE)
                # Since the stats are asynchronously pushed from the GPU to CPU, they are not always reliable
                # Therefore, a synchronization is required to guarantee you see the intended value.
                # Without a synchronization, it is possible for some GPUs to go through the exit conditional
                # and others to not because they accidentally see a different value for `skipped_optimizer_steps`.
                # In order to remove most device syncs, synchronizations only begin in the last few steps
                # where the skipped step count matters.

                # print(f"Outer, step: {step}, static_optimizer_step: {static_optimizer_step}, global_resume_step: {global_resume_step}, args.steps_this_run: {args.steps_this_run}, timeout_sent: {timeout_sent}")
                if static_optimizer_step + global_resume_step >= args.steps_this_run or timeout_sent:
                    torch.cuda.synchronize()
                    dynamic_optimizer_step = static_optimizer_step - int(stats.host_stat_value('skipped_optimizer_steps')) + global_resume_step
                    # print(f"args.steps_this_run: {args.steps_this_run}, dynamic_optimizer_step: {dynamic_optimizer_step}, timeout_sent: {timeout_sent}")
                    if dynamic_optimizer_step >= args.steps_this_run or timeout_sent:
                        train_time_raw = time.time() - raw_train_start
                        # print("entered dynamic optimizer step")
                        last_num_steps = args.log_freq if no_log_steps == 0 else no_log_steps
                        stats.device_stat('average_loss').div_(last_num_steps * args.gradient_accumulation_steps)
                        if (torch.distributed.is_initialized()):
                            stats.device_stat('average_loss').div_(get_world_size())
                            print("DEBUG: all_reduce ongoing")
                            torch.distributed.all_reduce(stats.device_stat('average_loss'))
                        
                        # We block on this copy to insure the final value
                        stats.host_stat('average_loss').copy_(stats.device_stat('average_loss'))
                        if is_main_process():
                            dllogger.log(step=(epoch, dynamic_optimizer_step,), data={"final_loss": stats.host_stat_value('average_loss')})

                        # checkpoint_only_to_cpu_step(args, epoch, dynamic_optimizer_step, model, optimizer, grad_scaler, most_recent_ckpts_paths, buffer_states, model_state_keys, optimizer_state_keys)

                        # Hari remove checkpoint step for single GPU training
                        # checkpoint_step(args, epoch, dynamic_optimizer_step, model, optimizer, grad_scaler, most_recent_ckpts_paths)

                        return args, train_time_raw, stats, skip_fwd_bwd_for_perf

                # load_checkpoints_from_disk(device, model, optimizer, grad_scaler, model_state_keys, optimizer_state_keys)

                if no_log_steps == 0:
                    if is_main_process():
                        dllogger.log(step=(epoch, dynamic_optimizer_step,),
                                     data={"average_loss": stats.host_stat_value('average_loss') / (args.log_freq * args.gradient_accumulation_steps),
                                           "learning_rate": stats.host_stat_value('learning_rate'),
                                           "skipped_steps": int(stats.host_stat_value('skipped_optimizer_steps'))})
                        if stats.host_stat_value('skip_optimizer_step') > 0.:
                            dllogger.log(step="PARAMETER", data={"loss_scale": grad_scaler._get_scale_async().item()})

                    stats.device_stat('average_loss').zero_()

                    # Hari remove checkpoint step for single GPU training
                    # if not args.skip_checkpoint and (dynamic_optimizer_step % args.num_steps_per_checkpoint == 0):
                    #     checkpoint_step(args, epoch, dynamic_optimizer_step, model, optimizer, grad_scaler, most_recent_ckpts_paths)


        epoch += 1

if __name__ == "__main__":
    now = time.time()
    args, train_time_raw, stats, skip_fwd_bwd_for_perf = main()
    gpu_count = args.n_gpu
    if torch.distributed.is_initialized():
        gpu_count = get_world_size()
    if is_main_process():
        e2e_time = time.time() - now
        training_perf = args.train_batch_size * gpu_count * (stats.host_stat_value('model_step') - skip_fwd_bwd_for_perf) / train_time_raw
        dllogger.log(step=tuple(), data={"e2e_train_time": e2e_time,
                                         "training_sequences_per_second": training_perf,
                                         "final_loss": stats.host_stat_value('average_loss'),
                                         "raw_train_time": train_time_raw })
    dllogger.flush()
