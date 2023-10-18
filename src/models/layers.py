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
date     : Sep  8, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : layers.py
project  : lab_exp_2023f
license  : GPL-3.0+

Layers
"""
from collections.abc import Sequence
from dataclasses import field

import jax
import jax.numpy as jnp
from flax import linen

PAD_ = str | int | Sequence[int | tuple[int, int]]


class DoubleConv(linen.Module):
  """Double Convolution Layer."""

  out_channels: int
  mid_channel: int | None = None
  kernel_size: int = 3
  stride: int = 1
  padding: PAD_ = "SAME"
  with_bn: bool = True
  bn_config: dict = field(default_factory=lambda: {
      "momentum": 0.99,
      "epsilon": 1e-5,
      "use_scale": True,
      "use_bias": True,
  })

  @linen.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    """Call."""
    mid_channel = self.mid_channel or self.out_channels

    x = linen.Conv(
        mid_channel,
        (self.kernel_size, self.kernel_size),
        self.stride,
        self.padding,
    )(
        x)
    if self.with_bn:
      x = linen.BatchNorm(**self.bn_config)(x, use_running_average=not train)
    x = jax.nn.relu(x)

    x = linen.Conv(
        self.out_channels,
        (self.kernel_size, self.kernel_size),
        self.stride,
        self.padding,
    )(
        x)
    if self.with_bn:
      x = linen.BatchNorm(**self.bn_config)(x, use_running_average=not train)
    return jax.nn.relu(x)


class Down(linen.Module):
  """Down Sampling Layer."""

  out_channels: int
  dropout: bool = True

  @linen.compact
  def __call__(self, x: jax.Array, train: bool = True) -> jax.numpy.ndarray:
    """Call."""
    x = linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
    if self.dropout:
      x = linen.Dropout(rate=0.5)(x, deterministic=not train)
    return DoubleConv(out_channels=self.out_channels)(x, train=train)


class UpSample(linen.Module):
  """Up Sampling Layer."""

  out_channels: int
  kernel_size: int = 2
  stride: int = 2
  padding: PAD_ = "SAME"
  dropout: bool = True

  @linen.compact
  def __call__(self, x1: jax.Array, x2: jax.Array, train: bool = True) -> jax.Array:
    """Call."""
    x1 = linen.ConvTranspose(
        self.out_channels,
        (self.kernel_size, self.kernel_size),
        (self.stride, self.stride),
        self.padding,
    )(
        x1)
    diff1 = x2.shape[1] - x1.shape[1]
    diff2 = x2.shape[2] - x1.shape[2]
    x1 = jnp.pad(
        x1,
        [
            (0, 0),
            (diff1 // 2, diff1 - diff1 // 2),
            (diff2 // 2, diff2 - diff2 // 2),
            (0, 0),
        ],
    )
    x = jnp.concatenate([x1, x2], axis=-1)
    if self.dropout:
      x1 = linen.Dropout(rate=0.5)(x1, deterministic=not train)
    return DoubleConv(out_channels=self.out_channels)(x, train=train)
