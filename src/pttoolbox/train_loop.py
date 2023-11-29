"""Simple (base) pytorch train loop."""
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
from accelerate import Accelerator
from rich.progress import Progress
from timm.utils.metrics import AverageMeter, accuracy


@dataclass
class Cfg:
    """Train loop config."""

    epochs: int = 5
    lr: float = 0.001
    # betas: tuple[float, float] = (0.95, 0.95)
    opt_cfg: dict[str, Any] = field(default_factory=dict)


def train_loop(
    cfg: Cfg,
    model: torch.nn.Module,
    opt_func: Callable,
    loss_func: Callable,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
):
    """Train loop."""
    opt = opt_func(model.parameters(), lr=cfg.lr, **cfg.opt_cfg)
    accelerator = Accelerator()
    model, opt, train_loader, val_loader, loss_func = accelerator.prepare(
        model, opt, train_loader, val_loader, loss_func
    )

    num_last = 10
    metrics = {
        "loss": torch.tensor(0.0, device=accelerator.device),
        "out": 0,
        "targets": torch.tensor(0.0, device=accelerator.device),
        "accuracy": torch.tensor(0.0, device=accelerator.device),
        "last_loss": 0,
        "losses": [],
        "val_losses": [],
        "train_acc": [],
        "val_acc": [],
        "last_losses": [1.0] * num_last,
        "acc_val": AverageMeter(),
        "acc_train": AverageMeter(),
    }

    def one_batch(xb: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        metrics["out"] = model(xb[0])
        metrics["loss"] = loss_func(metrics["out"], xb[1])
        metrics["targets"] = xb[1]
        metrics["accuracy"] = accuracy(metrics["out"], metrics["targets"])[0]
        return metrics

    def result_reset():
        metrics["acc_train"].reset()
        metrics["acc_val"].reset()
        metrics["val_loss"] = torch.tensor(0.0, device=accelerator.device)

    def record_batch(mode: Literal["train", "validate"]):
        if mode == "train":
            metrics["acc_train"].update(metrics["accuracy"].mean().cpu().item())
            metrics["last_losses"].pop(0)
            metrics["last_losses"].append(metrics["loss"].mean().cpu().item())
            metrics["last_loss"] = sum(metrics["last_losses"]) / num_last
        elif mode == "validate":
            metrics["val_loss"].add_(metrics["loss"].sum())
            metrics["acc_val"].update(
                metrics["accuracy"].item(), metrics["targets"].size(0)
            )

    def record_epoch(mode: Literal["train", "validate"]):
        if mode == "train":
            metrics["losses"].append(metrics["last_loss"])
            metrics["train_acc"].append(metrics["acc_train"].avg)
        elif mode == "validate":
            metrics["val_losses"].append(
                metrics["val_loss"].sum().item() / len(val_loader.dataset)
            )
            metrics["val_acc"].append(metrics["acc_val"].avg)

    def print_epoch_result(epoch: int):
        print(
            f"epoch {epoch + 1} {metrics['train_acc'][-1] / 100:0.2%} {metrics['losses'][-1]:0.3f}    {metrics['val_acc'][-1] / 100:0.2%} {metrics['val_losses'][-1]:0.3f}"
        )

    with Progress(transient=True) as progress:
        main_task = progress.add_task("", total=cfg.epochs)
        for epoch in range(cfg.epochs):
            progress.tasks[main_task].description = f"Epoch {epoch + 1} / {cfg.epochs}"

            result_reset()
            # train
            model.train()

            train_task = progress.add_task("Train", total=len(train_loader))
            for xb in train_loader:
                one_batch(xb)
                accelerator.backward(metrics["loss"].sum())
                opt.step()
                for param in model.parameters():
                    param.grad = None
                record_batch("train")
                progress.update(train_task, advance=1)
                progress._tasks[
                    train_task
                ].description = f"loss: {metrics['last_loss']:0.4f}"

            record_epoch("train")
            progress._tasks[
                train_task
            ].description = (
                f"{metrics['train_acc'][-1] / 100:.2%} {metrics['last_loss']:0.3f}"
            )

            # validate
            with torch.no_grad():
                model.eval()
                val_task = progress.add_task("Val", total=len(val_loader))
                for xb in val_loader:
                    one_batch(xb)
                    record_batch("validate")
                    progress.update(val_task, advance=1)

            record_epoch("validate")
            print_epoch_result(epoch)

            progress._tasks.pop(val_task)
            progress._tasks.pop(train_task)
            progress.update(main_task, advance=1)

        progress.remove_task(main_task)

    return metrics


def print_result(metrics: dict[str, torch.Tensor]):
    for item in zip(
        range(1, len(metrics["train_acc"]) + 1),
        metrics["train_acc"],
        metrics["losses"],
        metrics["val_acc"],
        metrics["val_losses"],
    ):
        print(
            f"ep {item[0]} {item[1] / 100:0.2%} {item[2]:0.3f}    {item[3] / 100:0.2%} {item[4]:0.3f}"
        )
