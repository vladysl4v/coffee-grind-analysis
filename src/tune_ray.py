import os
import tempfile

import ray
import torch
from ray import tune
import torch.nn as nn
import torch.optim as optim
from ray.tune.schedulers import ASHAScheduler
from torch.amp import autocast, GradScaler

from train import _fgsm_perturb

import numpy as np

from data_loader import get_loaders, DEFAULT_TRAIN_TRANSFORM, DEFAULT_EVAL_TRANSFORM, _NORMALIZE, _RAW_IMAGES_DIR
from augmentation import ApplyAugmentation, seed_worker
from torchvision import transforms as T
from models.fpn_resnet import FPNResnet18
from pathlib import Path

def create_dataloaders(batch_size, num_workers):
    train_loader, val_loader, _ = get_loaders(
        batch_size=batch_size,
        train_transform=DEFAULT_TRAIN_TRANSFORM,
        num_workers=num_workers,
        use_augmented_data=False,
        use_augmented_raw=False,
        use_raw=False,
        worker_init_fn=None,
    )
    return train_loader, val_loader

def train_epoch(config):
    assert torch.cuda.is_available()
    net = FPNResnet18()
    device = config["device"]
    net.to(device)

    criterion = nn.HuberLoss(delta=config["huber_delta"])
    optimizer = optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scaler = GradScaler(device="cuda")
    if tune.get_checkpoint():
        loaded_checkpoint = tune.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    train_loader, val_loader = create_dataloaders(config["batch_size"], config["num_workers"])
    acc_mse = torch.zeros(1, device=device)
    acc_mae = torch.zeros(1, device=device)
    for epoch in range(config["max_num_epochs"]):
        net.train()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            images_adv = _fgsm_perturb(images, labels, net, criterion, config["adv_epsilon"], device)
            optimizer.zero_grad()

            with autocast(device_type=device):
                preds_clean = net(images).view(-1)
                loss_clean = criterion(preds_clean, labels)
                preds_adv = net(images_adv).view(-1)
                loss_adv = criterion(preds_adv, labels)
            loss = (1 - config["adv_weight"]) * loss_clean + config["adv_weight"] * loss_adv
            preds = preds_clean

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                acc_mse += loss.detach()
                acc_mae += (preds.detach() - labels).abs().mean()

        acc_mse = torch.zeros(1, device=device)
        acc_mae = torch.zeros(1, device=device)
        net.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast(device_type=device):
                    preds = net(images).view(-1)
                    acc_mse += criterion(preds, labels)
                acc_mae += (preds - labels).abs().mean()

        metrics = {
            "acc_mae": acc_mae.item() / len(val_loader),
            "acc_mse": acc_mse.item() / len(val_loader),
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
            tune.report(metrics, checkpoint=checkpoint)

        print("Finished")

def test_best_model(best_result, smoke_test=False):
    best_trained_model = FPNResnet18()
    device = best_result.config["device"]
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, _optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    _, _, testloader = get_loaders(
        batch_size=best_result.config["batch_size"],
        num_workers=0)

    criterion = nn.HuberLoss(delta=best_result.config["huber_delta"])
    acc_mse = torch.zeros(1, device=device)
    acc_mae = torch.zeros(1, device=device)
    best_trained_model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device_type=device):
                preds = best_trained_model(images).view(-1)
                acc_mse += criterion(preds, labels)
            acc_mae += (preds - labels).abs().mean()

    print(f"Best values: mse: {acc_mse / len(testloader)} mae: {acc_mae / len(testloader)}")

config = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([16, 32]),
    "adv_epsilon": tune.uniform(0.005, 0.2),
    "adv_weight": tune.uniform(0.4, 0.6),
    "huber_delta": tune.uniform(0.01, 0.3),
    "num_workers": 8,
    "max_num_epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def main(config, gpus_per_trial=1):
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config["max_num_epochs"],
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_epoch),
            resources={"cpu": 8, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="acc_mae",
            mode="min",
            scheduler=scheduler,
            num_samples=20,
            max_concurrent_trials=0,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("acc_mae", "min")

    print(f"Best trial config: {best_result.config}")
    print(best_result.metrics)
    #print(f"Best trial final validation loss: {best_result.metrics['acc_mse']}")
    #print(f"Best trial final validation accuracy: {best_result.metrics['acc_mae']}")

    test_best_model(best_result)


main(config, gpus_per_trial=1 if torch.cuda.is_available() else 0)



