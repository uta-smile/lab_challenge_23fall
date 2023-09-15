#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Python ♡ Nasy.

    |             *         *
    |                  .                .
    |           .                              登
    |     *                      ,
    |                   .                      至
    |
    |                               *          恖
    |          |\___/|
    |          )    -(             .           聖 ·
    |         =\ -   /=
    |           )===(       *
    |          /   - \
    |          |-    |
    |         /   -   \     0.|.0
    |  NASY___\__( (__/_____(\=/)__+1s____________
    |  ______|____) )______|______|______|______|_
    |  ___|______( (____|______|______|______|____
    |  ______|____\_|______|______|______|______|_
    |  ___|______|______|______|______|______|____
    |  ______|______|______|______|______|______|_
    |  ___|______|______|______|______|______|____

author   : Nasy https://nasy.moe
date     : Sep 10, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : data.py
project  : lab_exp_2023f
license  : GPL-3.0+

Dataset and Loader.
"""

from collections.abc import Sequence, Iterator

from config import Conf
from .input import read_images
import jax
import jax.numpy as jnp
import haiku as hk
from operator import itemgetter

from dataclasses import dataclass


@dataclass
class Loader:
  """Data Loader."""

  data: Sequence[tuple[jax.Array, jax.Array]]
  conf: Conf
  batch_size: int = 2
  shuffle: bool = False
  rngs: hk.PRNGSequence | None = None

  def __len__(self) -> int:
    """Length."""
    return len(self.data) // self.batch_size

  def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
    """Iterate."""
    data = self.data
    if self.shuffle:
      if self.rngs is None:
        self.rngs = hk.PRNGSequence(42)
      rng = next(self.rngs)
      rarr = jax.random.permutation(rng, jnp.arange(len(self.data)))
      data = itemgetter(*rarr)(self.data)

    for i in range(len(self)):
      xs, ys = zip(*data[i * self.batch_size : (i + 1) * self.batch_size], strict=True)
      yield jnp.stack(xs), jnp.stack(ys)


def test_loader() -> None:
  """Test Loader."""
  conf = Conf()
  data = read_images(conf.data.base / "mnseg")
  loader = Loader(data, conf)
  for i, (x, y) in enumerate(loader):
    print(i, x.shape, y.shape)


if __name__ == "__main__":
  test_loader()
