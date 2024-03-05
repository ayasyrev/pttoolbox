"""Samplers/ First - persistent sampler."""

from typing import Iterator, Optional

from torch.utils.data import Sampler


class PersistentSampler(Sampler[int]):
    def __init__(
        self,
        indexes: list[list[int]],
        epochs: Optional[int] = None,
    ) -> None:
        self.indexes = indexes
        self.epochs = epochs or len(indexes)
        self._num_samples = len(self.indexes[0])
        for epoch in range(1, self.epochs):
            assert len(self.indexes[epoch]) == self._num_samples
        self.epoch = 0

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        yield from self.indexes[self.epoch]
        self.epoch += 1
        if self.epoch == self.epochs:
            self.epoch = 0
