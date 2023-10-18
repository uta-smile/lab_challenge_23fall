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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from config import Conf

from .input import read_images


@dataclass
class Embed:
  """Embedding."""

  im: Sequence[jax.Array]
  mask: Sequence[jax.Array]

  def __post_init__(self) -> None:
    """Post Init."""
    self.ima = jnp.stack(self.im)
    self.maska = jnp.stack(self.mask)

  def embed(self, idx: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Embed."""
    return self.ima[idx], self.maska[idx]


@dataclass
class Loader:
  """Data Loader."""

  length: int
  batch_size: int = 2
  rng: jax.Array | int = field(default_factory=lambda: jax.random.key(0))
  shuffle: bool = False
  droplast: bool = False
  rebuild: int = 1

  def __post_init__(self) -> None:
    """Post Init."""
    self.__rebuild = self.rebuild
    self.build()

  def build(self) -> None:
    """Build."""
    if self.shuffle:
      if isinstance(self.rng, int):
        self.rng = jax.random.key(self.rng)
      self.rng, rng = jax.random.split(self.rng)
      self.idx = jax.random.permutation(rng, self.length)
    else:
      self.idx = jnp.arange(self.length)

    if self.droplast:
      self.idx = self.idx[:self.length - self.length % self.batch_size]
    elif self.length % self.batch_size:
      self.idx = jnp.concatenate(
          [self.idx, self.idx[:self.batch_size - self.length % self.batch_size]])
    self.idx = self.idx.reshape(-1, self.batch_size)
    self.rebuild = self.__rebuild

  def __iter__(self) -> Iterator[jax.Array]:
    """Run Loader."""
    if not self.rebuild:
      self.build()
    self.rebuild -= 1
    yield from self.idx

  def __len__(self) -> int:
    """Length."""
    return self.length // self.batch_size


@dataclass
class Data:
  """Data."""

  tr: Loader
  te: Loader
  tr_embed: Embed
  te_embed: Embed


def build(conf: Conf, have_mask: bool = False) -> Data:
  """Build dataset."""
  tr_im = read_images(conf.data.base / "mnseg" / "im")
  tr_mask = read_images(conf.data.base / "mnseg" / "mask", binary=True)
  te_im = read_images(conf.data.base / "mnseg_test" / "im")
  if have_mask:
    te_mask = read_images(conf.data.base / "mnseg_test" / "mask", binary=True)
  else:
    # we don't have mask, so, regard im as mask to keep the same input as train.
    te_mask = read_images(conf.data.base / "mnseg_test" / "im", binary=True)

  return Data(
      Loader(len(tr_im), batch_size=conf.params.batch_size, shuffle=True),
      Loader(len(te_im), batch_size=1, shuffle=False),
      Embed(tr_im, tr_mask),
      Embed(te_im, te_mask),
  )


def test() -> None:
  """Test data."""
  conf = Conf()
  data = build(conf)
  print("Train Loader:")
  for tr in data.tr:
    print(data.tr_embed.embed(tr)[0].shape)
    print(data.tr_embed.embed(tr)[1].shape)
  print("Test Loader:")
  for te in data.te:
    print(data.te_embed.embed(te)[0].shape)


if __name__ == "__main__":
  test()
