"""Simple (base) pytorch train loop."""
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import torch
from accelerate import Accelerator
from rich import print as rprint
from rich.progress import Progress
from timm.utils.metrics import AverageMeter, accuracy


@dataclass
class Cfg:
    """Train loop config."""

    epochs: int = 5
    lr: float = 0.001
    opt_cfg: dict[str, Any] = field(default_factory=dict)


def train_loop(
    cfg: Cfg,
    model: torch.nn.Module,
    opt_func: Callable,
    loss_func: Callable,
    dl_train: torch.utils.data.DataLoader,
    dl_validate: torch.utils.data.DataLoader,
    batch_transform: Optional[Callable] = None,
):
    """Train loop."""
    opt = opt_func(model.parameters(), lr=cfg.lr, **cfg.opt_cfg)
    accelerator = Accelerator()
    model, opt, dl_train, dl_validate, loss_func = accelerator.prepare(
        model, opt, dl_train, dl_validate, loss_func
    )

    num_last = 10
    metrics = {
        # batch level
        "loss": torch.tensor(0.0, device=accelerator.device),
        "out": 0,
        "targets": torch.tensor(0.0, device=accelerator.device),
        "accuracy": torch.tensor(0.0, device=accelerator.device),

        # last `num_last` train losses for calculating average
        "loss_train": [1.0] * num_last,
        "loss_validate": torch.tensor(0.0, device=accelerator.device),

        # epoch level
        "acc_avgmeter_train": AverageMeter(),
        "acc_avgmeter_validate": AverageMeter(),

        "losses_train": [],
        "losses_validate": [],
        "accuracy_train": [],
        "accuracy_validate": [],

        "time_train": [],
        "time_validate": [],
    }

    if batch_transform:
        def one_batch(xb: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
            metrics["out"] = model(batch_transform(xb[0]))
            metrics["loss"] = loss_func(metrics["out"], xb[1])
            metrics["targets"] = xb[1]
            metrics["accuracy"] = accuracy(metrics["out"], metrics["targets"])[0]
            return metrics
    else:
        def one_batch(xb: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
            metrics["out"] = model(xb[0])
            metrics["loss"] = loss_func(metrics["out"], xb[1])
            metrics["targets"] = xb[1]
            metrics["accuracy"] = accuracy(metrics["out"], metrics["targets"])[0]
            return metrics

    def result_reset():
        metrics["acc_avgmeter_train"].reset()
        metrics["acc_avgmeter_validate"].reset()
        metrics["loss_validate"] = torch.tensor(0.0, device=accelerator.device)

    def record_batch(mode: Literal["train", "validate"]):
        if mode == "train":
            metrics["acc_avgmeter_train"].update(metrics["accuracy"].mean().cpu().item())
            metrics["loss_train"].pop(0)
            metrics["loss_train"].append(metrics["loss"].mean().cpu().item())
        elif mode == "validate":
            metrics["loss_validate"].add_(metrics["loss"].sum())
            metrics["acc_avgmeter_validate"].update(
                metrics["accuracy"].item(), metrics["targets"].size(0)
            )

    def record_epoch(mode: Literal["train", "validate"]):
        if mode == "train":
            metrics["losses_train"].append(sum(metrics["loss_train"]) / num_last)
            metrics["accuracy_train"].append(metrics["acc_avgmeter_train"].avg)
        elif mode == "validate":
            metrics["losses_validate"].append(
                metrics["loss_validate"].sum().item() / len(dl_validate.dataset)
            )
            metrics["accuracy_validate"].append(metrics["acc_avgmeter_validate"].avg)

    def print_epoch_result(epoch: int):
        rprint(
            f"epoch {epoch + 1} {metrics['accuracy_train'][-1] / 100:0.2%} {metrics['losses_train'][-1]:0.3f}"
            f"    {metrics['accuracy_validate'][-1] / 100:0.2%} {metrics['losses_validate'][-1]:0.3f}"
            f"    {metrics['time_train'][-1]:0.1f} / {metrics['time_validate'][-1]:0.1f} sec"
        )

    with Progress(transient=True) as progress:
        main_task = progress.add_task("", total=cfg.epochs)
        for epoch in range(cfg.epochs):
            progress.tasks[main_task].description = f"Epoch {epoch + 1} / {cfg.epochs}"
            result_reset()

            # train
            model.train()
            train_task = progress.add_task("Train", total=len(dl_train))
            for xb in dl_train:
                one_batch(xb)
                accelerator.backward(metrics["loss"].sum())
                opt.step()
                for param in model.parameters():
                    param.grad = None
                record_batch("train")
                progress.update(train_task, advance=1)
                progress._tasks[
                    train_task
                ].description = f"loss: {sum(metrics['loss_train']) / num_last:0.4f}"

            metrics["time_train"].append(progress._tasks[train_task].finished_time)
            record_epoch("train")

            # validate
            with torch.no_grad():
                model.eval()
                val_task = progress.add_task("Val", total=len(dl_validate))
                for xb in dl_validate:
                    one_batch(xb)
                    record_batch("validate")
                    progress.update(val_task, advance=1)
            metrics["time_validate"].append(progress._tasks[val_task].finished_time)
            record_epoch("validate")
            print_epoch_result(epoch)

            progress._tasks.pop(val_task)
            progress._tasks.pop(train_task)
            progress.update(main_task, advance=1)

        progress.remove_task(main_task)
    return metrics
