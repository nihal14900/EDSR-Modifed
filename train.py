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
# ============================================================================
"""File description: Realize the model training function."""
import os
import shutil
import time
from enum import Enum

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import imgproc
from dataset import CUDAPrefetcher
from dataset import TrainValidImageDataset, TestImageDataset
from models.model15 import EDSR


def main():
    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0

    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset()
    print("Load train dataset and valid dataset successfully.")

    model = build_model()
    print("Build EDSR model successfully.")

    psnr_criterion, pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        config.start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        if checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    for epoch in range(config.start_epoch, config.epochs):
        train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)
        _ = validate(model, valid_prefetcher, psnr_criterion, epoch, writer, "Valid")
        psnr = validate(model, test_prefetcher, psnr_criterion, epoch, writer, "Test")
        print("\n")

        # Update lr
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save({"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"), os.path.join(results_dir, "best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"), os.path.join(results_dir, "last.pth.tar"))


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_image_dir, config.image_size, config.upscale_factor, "Train")
    valid_datasets = TrainValidImageDataset(config.valid_image_dir, config.image_size, config.upscale_factor, "Valid")
    test_datasets = TestImageDataset(config.test_lr_image_dir, config.test_hr_image_dir, config.upscale_factor)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, valid_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    model = EDSR(config.upscale_factor).to(config.device)

    return model


def define_loss() -> [nn.MSELoss, nn.L1Loss]:
    psnr_criterion = nn.MSELoss().to(config.device)
    pixel_criterion = nn.L1Loss().to(config.device)

    return psnr_criterion, pixel_criterion


def define_optimizer(model) -> optim.Adam:
    # optimizer = optim.Adam(model.parameters(), lr=config.model_lr, betas=config.model_betas)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)

    # model15-ca-norestart
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=3e-4)

    # model25
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.MultiStepLR:
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.lr_scheduler_gamma)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-4)

    # model15-ca-norestart
    # scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-4)  # T_max=500 ensures smooth decay

    # model25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)

    return scheduler


def train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer) -> None:
    # Calculate how many iterations there are under epoch
    batches = len(train_prefetcher)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses, psnres], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator in training mode
    model.train()

    batch_index = 0

    # Calculate the time it takes to test a batch of data
    end = time.time()
    # enable preload
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    while batch_data is not None:
        # measure data loading time
        data_time.update(time.time() - end)

        lr = batch_data["lr"].to(config.device, non_blocking=True)
        hr = batch_data["hr"].to(config.device, non_blocking=True)

        # Initialize the generator gradient
        model.zero_grad()

        # Mixed precision training
        with amp.autocast():
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        # Gradient zoom
        scaler.scale(loss).backward()

        # Apply Gradient Clipping before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update generator weight
        scaler.step(optimizer)
        scaler.update()

        # ========================= ADDED CODE =========================
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]  

        # Print learning rate
        if batch_index % config.print_frequency == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}], Batch [{batch_index}/{batches}], LR: {current_lr:.8f}")

        # Log learning rate to TensorBoard
        writer.add_scalar("Train/Learning Rate", current_lr, batch_index + epoch * batches + 1)
        # =============================================================

        # measure accuracy and record loss
        psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
        losses.update(loss.item(), lr.size(0))
        psnres.update(psnr.item(), lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record training log information
        if batch_index % config.print_frequency == 0:
            # Writer Loss to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After a batch of data is calculated, add 1 to the number of batches
        batch_index += 1


def validate(model, valid_prefetcher, psnr_criterion, epoch, writer, mode) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(len(valid_prefetcher), [batch_time, psnres], prefix=f"{mode}: ")

    # Put the model in verification mode
    model.eval()

    batch_index = 0

    # Calculate the time it takes to test a batch of data
    end = time.time()
    with torch.no_grad():
        # enable preload
        valid_prefetcher.reset()
        batch_data = valid_prefetcher.next()

        while batch_data is not None:
            # measure data loading time
            lr = batch_data["lr"].to(config.device, non_blocking=True)
            hr = batch_data["hr"].to(config.device, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # Convert RGB tensor to Y tensor
            sr_image = imgproc.tensor2image(sr, range_norm=False, half=True)
            sr_image = sr_image.astype(np.float32) / 255.
            sr_y_image = imgproc.rgb2ycbcr(sr_image, use_y_channel=True)
            sr_y_tensor = imgproc.image2tensor(sr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

            hr_image = imgproc.tensor2image(hr, range_norm=False, half=True)
            hr_image = hr_image.astype(np.float32) / 255.
            hr_y_image = imgproc.rgb2ycbcr(hr_image, use_y_channel=True)
            hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

            # Define crop border size based on upscale factor
            crop_border = config.upscale_factor

            # Crop border before computing PSNR
            sr_y_image = sr_y_image[crop_border:-crop_border, crop_border:-crop_border]
            hr_y_image = hr_y_image[crop_border:-crop_border, crop_border:-crop_border]

            # measure accuracy and record loss
            psnr = 10. * torch.log10(1. / psnr_criterion(sr_y_tensor, hr_y_tensor))
            psnres.update(psnr.item(), lr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % config.print_frequency == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = valid_prefetcher.next()

            # After a batch of data is calculated, add 1 to the number of batches
            batch_index += 1

    # Print average PSNR metrics
    progress.display_summary()

    if mode == "Valid":
        writer.add_scalar("Valid/PSNR", psnres.avg, epoch + 1)
    elif mode == "Test":
        writer.add_scalar("Test/PSNR", psnres.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg


# Copy form "https://github.com/pytorch/examples/blob/master/imagenet/main.py"
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
