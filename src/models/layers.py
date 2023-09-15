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
from dataclasses import dataclass, field

import haiku as hk
import jax
import jax.numpy as jnp

PAD_ = str | Sequence[tuple[int, int]] | hk.pad.PadFn | Sequence[hk.pad.PadFn]


@dataclass
class DoubleConv(hk.Module):
  """Double Convolution Layer."""

  out_channels: int
  mid_channel: int | None = None
  kernel_size: int = 3
  stride: int = 1
  padding: PAD_ = "SAME"
  with_bn: bool = True
  bn_config: dict = field(
      default_factory=lambda: {
          "decay_rate": 0.99,
          "eps": 1e-4,
          "create_scale": True,
          "create_offset": True,
      }
  )

  def __call__(self, x: jax.Array, train: bool = True) -> jax.numpy.ndarray:
    """Call."""
    if self.mid_channel is None:
      self.mid_channel = self.out_channels

    x = hk.Conv2D(  # type: ignore
        output_channels=self.out_channels,
        kernel_shape=self.kernel_size,
        stride=self.stride,
        padding=self.padding,
        with_bias=True,
    )(x)
    if self.with_bn:
      x = hk.BatchNorm(**self.bn_config)(x, is_training=train)  # type: ignore
    x = jax.nn.relu(x)

    x = hk.Conv2D(  # type: ignore
        output_channels=self.out_channels,
        kernel_shape=self.kernel_size,
        stride=self.stride,
        padding=self.padding,
        with_bias=True,
    )(x)
    if self.with_bn:
      x = hk.BatchNorm(**self.bn_config)(x, is_training=train)  # type: ignore
    return jax.nn.relu(x)


@dataclass
class Down(hk.Module):
  """Down Sampling Layer."""

  out_channels: int

  def __call__(self, x: jax.Array, train: bool = True) -> jax.numpy.ndarray:
    """Call."""
    x = hk.max_pool(x, window_shape=2, strides=2, padding="VALID")
    return DoubleConv(out_channels=self.out_channels)(x, train=train)  # type: ignore


@dataclass
class UpSample(hk.Module):
  """Up Sampling Layer."""

  out_channels: int
  kernel_size: int = 2
  stride: int = 2
  padding: str = "SAME"

  def __call__(
      self, x1: jax.Array, x2: jax.Array, train: bool = True
  ) -> jax.numpy.ndarray:
    """Call."""
    x1 = hk.Conv2DTranspose(  # type: ignore
        output_channels=self.out_channels,
        kernel_shape=self.kernel_size,
        stride=self.stride,
        padding=self.padding,
        with_bias=True,
    )(x1)
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
    return DoubleConv(out_channels=self.out_channels)(x, train=train)  # type: ignore
