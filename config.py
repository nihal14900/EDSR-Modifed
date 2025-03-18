# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = False
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "model15-df2k"

if mode == "train":
    # Dataset
    train_image_dir = r"/kaggle/input/df2kdata/DF2K_train_HR"
    valid_image_dir = r"/kaggle/input/df2kdata/DF2K_valid_HR"
    test_lr_image_dir = r"/kaggle/input/super-resolution-benchmarks/Set5/Set5/LRbicx4"
    test_hr_image_dir = r"/kaggle/input/super-resolution-benchmarks/Set5/Set5/GTmod12"

    image_size = int(upscale_factor * 64)
    batch_size = 16
    num_workers = 8

    # Incremental training and migration training
    start_epoch = 0
    resume = r"/kaggle/input/model15-df2k-trained206/pytorch/default/1/epoch_206.pth.tar"

    # Total num epochs
    epochs = 1000

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    # StepLR scheduler parameter
    lr_scheduler_step_size = 20
    lr_scheduler_gamma = 0.8

    print_frequency = 1000

if mode == "valid":
    # Test data address
    sr_dir = f"results/test/{exp_name}"

    lr_dir = r"E:\Classical\Set5\LRbicx4"
    hr_dir = r"E:\Classical\Set5\GTmod12"

    # lr_dir = r"E:\Classical\Set14\LRbicx4"
    # hr_dir = r"E:\Classical\Set14\GTmod12"

    # lr_dir = r"E:\Classical\manga109\LR\x4"
    # hr_dir = r"E:\Classical\manga109\GTmod12"

    # lr_dir = r"E:\Benchmark\B100\LR\x4"
    # hr_dir = r"E:\Benchmark\B100\GTmod12"

    # lr_dir = r"E:\Benchmark\Urban100\LR\x4"
    # hr_dir = r"E:\Benchmark\Urban100\GTmod12"

    model_path = f"results/{exp_name}/best.pth.tar"
