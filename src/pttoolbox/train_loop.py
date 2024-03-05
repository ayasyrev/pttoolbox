"""Simple (base) pytorch train loop."""

import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict, Union

import torch
import yaml
from accelerate import Accelerator
from rich import print as rprint
from rich.progress import Progress
from timm.utils.metrics import AverageMeter, accuracy


@dataclass
class Cfg:
    """Train loop config."""

    exp_name: str = ""
    ds: str = ""
    model_name: str = ""
    model_weights: Optional[str] = None
    epochs: int = 5
    lr: float = 0.001
    opt_func: type[torch.optim.Optimizer] = torch.optim.AdamW
    opt_cfg: dict[str, Any] = field(default_factory=dict)
    loss_func: type[torch.nn.Module] = torch.nn.CrossEntropyLoss
    loss_func_cfg: dict[str, Any] = field(default_factory=lambda: {"reduction": "none"})
    batch_transform: str = "normalize"
    batch_size: int = 32
    log_path: Union[str, Path] = "."
    num_last: int = 10


class Metrics(TypedDict):
    """Train loop metrics."""

    loss: torch.Tensor
    out: int
    targets: torch.Tensor
    accuracy: torch.Tensor
    loss_train: list[float]
    loss_validate: torch.Tensor
    acc_avgmeter_train: AverageMeter
    acc_avgmeter_validate: AverageMeter
    losses_train: list[float]
    losses_validate: list[float]
    accuracy_train: list[float]
    accuracy_validate: list[float]
    time_train: list[float]
    time_validate: list[float]
    num_last: int
    dl_train_len: int
    dl_validate_len: int


def initiate_metrics(cfg: Optional[Cfg] = None, device=None) -> Metrics:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_last = 10 if cfg is None else cfg.num_last
    return {
        # batch level
        "loss": torch.tensor(0.0, device=device),
        "out": 0,
        "targets": torch.tensor(0.0, device=device),
        "accuracy": torch.tensor(0.0, device=device),
        "loss_train": [1.0] * num_last,
        "loss_validate": torch.tensor(0.0, device=device),
        # epoch level
        "acc_avgmeter_train": AverageMeter(),
        "acc_avgmeter_validate": AverageMeter(),
        "losses_train": [],
        "losses_validate": [],
        "accuracy_train": [],
        "accuracy_validate": [],
        "time_train": [],
        "time_validate": [],
        "num_last": num_last,
        "dl_train_len": 0,
        "dl_validate_len": 0,
    }


def metrics_reset(metrics: Metrics):
    metrics["acc_avgmeter_train"].reset()
    metrics["acc_avgmeter_validate"].reset()
    metrics["loss_validate"] = metrics["loss_validate"].zero_()


def record_batch(mode: Literal["train", "validate"], metrics: Metrics) -> None:
    if mode == "train":
        metrics["acc_avgmeter_train"].update(
            # metrics["accuracy"].mean().cpu().item()
            metrics["accuracy"]
        )
        metrics["loss_train"].pop(0)
        # metrics["loss_train"].append(metrics["loss"].mean().cpu().item())
        metrics["loss_train"].append(metrics["loss"].mean().cpu().item())
    elif mode == "validate":
        metrics["loss_validate"].add_(metrics["loss"].sum())
        metrics["acc_avgmeter_validate"].update(
            # metrics["accuracy"].item(), metrics["targets"].size(0)
            metrics["accuracy"],
            metrics["targets"].size(0),
        )


def record_epoch(mode: Literal["train", "validate"], metrics: Metrics) -> None:
    if mode == "train":
        metrics["losses_train"].append(sum(metrics["loss_train"]) / metrics["num_last"])
        metrics["accuracy_train"].append(metrics["acc_avgmeter_train"].avg.item())
    elif mode == "validate":
        metrics["losses_validate"].append(
            metrics["loss_validate"].sum().item() / metrics["dl_validate_len"]
        )
        metrics["accuracy_validate"].append(metrics["acc_avgmeter_validate"].avg.item())


def print_epoch_result(
    epoch: int, metrics: Metrics, to_log: list[str], log_result
) -> None:
    rprint(
        f"epoch {epoch + 1} {metrics['accuracy_train'][-1] / 100:0.2%} {metrics['losses_train'][-1]:0.3f}"
        f"    {metrics['accuracy_validate'][-1] / 100:0.2%} {metrics['losses_validate'][-1]:0.3f}"
        f"    {metrics['time_train'][-1]:0.1f} / {metrics['time_validate'][-1]:0.1f} sec"
    )
    log_result.write(", ".join(str(metrics[key][-1]) for key in to_log) + "\n")
    log_result.flush()
    os.fsync(log_result.fileno())


def train_loop(
    cfg: Cfg,
    model: torch.nn.Module,
    dl_train: torch.utils.data.DataLoader,
    dl_validate: torch.utils.data.DataLoader,
    batch_transform: Optional[Callable] = None,
    metrics: Optional[Metrics] = None,
) -> Metrics:
    """Train loop."""
    opt = cfg.opt_func(model.parameters(), lr=cfg.lr, **cfg.opt_cfg)
    loss_func = cfg.loss_func(**cfg.loss_func_cfg)
    accelerator = Accelerator()
    if metrics is None:
        metrics = initiate_metrics(cfg, accelerator.device.type)
    metrics["dl_train_len"] = len(dl_train.dataset)
    metrics["dl_validate_len"] = len(dl_validate.dataset)
    model, opt, dl_train, dl_validate, loss_func = accelerator.prepare(
        model, opt, dl_train, dl_validate, loss_func
    )

    log_path = Path(cfg.log_path) / "__".join(
        [datetime.now().strftime("%Y%m%d-%H%M%S"), cfg.exp_name]
    )
    log_path.mkdir(parents=True, exist_ok=True)

    with open(log_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(asdict(cfg), f, indent=4)
    with open(log_path / "model.txt", "w", encoding="utf-8") as f:
        f.write(str(model))
    headers = [
        "loss_train",
        "accuracy_train",
        "loss_validate",
        "accuracy_validate",
        "time_train",
        "time_validate",
    ]
    to_log = [
        "losses_train",
        "accuracy_train",
        "losses_validate",
        "accuracy_validate",
        "time_train",
        "time_validate",
    ]

    log_name = Path(cfg.log_path) / "log_result.csv"
    log_result = open(log_name, "w", encoding="utf-8")
    log_result.write(", ".join(headers) + "\n")

    if batch_transform:

        def one_batch(
            batch: tuple[torch.Tensor, torch.Tensor],
            metrics: Metrics,
        ) -> Metrics:
            batch[0] = batch[0].to(memory_format=torch.channels_last)
            metrics["out"] = model(batch_transform(batch[0]))
            metrics["loss"] = loss_func(metrics["out"], batch[1])
            metrics["targets"] = batch[1]
            metrics["accuracy"] = accuracy(metrics["out"], metrics["targets"])[0]
            return metrics

    else:

        def one_batch(
            batch: tuple[torch.Tensor, torch.Tensor],
            metrics: Metrics,
        ) -> Metrics:
            metrics["out"] = model(batch[0])
            metrics["loss"] = loss_func(metrics["out"], batch[1])
            metrics["targets"] = batch[1]
            metrics["accuracy"] = accuracy(metrics["out"], metrics["targets"])[0]
            return metrics

    def one_batch_val(
        batch: tuple[torch.Tensor, torch.Tensor],
        metrics: Metrics,
    ) -> Metrics:
        metrics["out"] = model(batch[0])
        metrics["loss"] = loss_func(metrics["out"], batch[1])
        metrics["targets"] = batch[1]
        metrics["accuracy"] = accuracy(metrics["out"], metrics["targets"])[0]
        return metrics

    model = model.to(memory_format=torch.channels_last)
    with Progress(transient=True) as progress:
        main_task = progress.add_task("", total=cfg.epochs)
        for epoch in range(cfg.epochs):
            progress.tasks[main_task].description = f"Epoch {epoch + 1} / {cfg.epochs}"
            metrics_reset(metrics)

            # train
            model.train()
            train_task = progress.add_task("Train", total=len(dl_train))
            for batch in dl_train:
                one_batch(batch, metrics)
                accelerator.backward(metrics["loss"].sum())
                opt.step()
                for param in model.parameters():
                    param.grad = None
                record_batch("train", metrics)
                progress.update(train_task, advance=1)
                # progress._tasks[
                #     train_task
                # ].description = f"loss: {sum(metrics['loss_train']) / num_last:0.4f}"
            if hasattr(dl_train.dataset, "step_epoch"):
                dl_train.dataset.step_epoch()
            metrics["time_train"].append(progress._tasks[train_task].finished_time)
            record_epoch("train", metrics)

            # validate
            with torch.no_grad():
                model.eval()
                val_task = progress.add_task("Val", total=len(dl_validate))
                for batch in dl_validate:
                    one_batch(batch, metrics)
                    # one_batch_val(batch, metrics)
                    record_batch("validate", metrics)
                    progress.update(val_task, advance=1)
            metrics["time_validate"].append(progress._tasks[val_task].finished_time)
            record_epoch("validate", metrics)
            print_epoch_result(epoch, metrics, to_log, log_result)

            progress._tasks.pop(val_task)
            progress._tasks.pop(train_task)
            progress.update(main_task, advance=1)

        progress.remove_task(main_task)
    log_result.close()
    shutil.copy(log_name, log_path / "log_result.csv")
    return metrics
