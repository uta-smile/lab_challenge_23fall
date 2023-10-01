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
filename : models.py
project  : lab_exp_2023f
license  : GPL-3.0+

Models
"""
from dataclasses import field

import jax
import jax.numpy as jnp
from flax import linen
from flax.traverse_util import flatten_dict

from .layers import DoubleConv, Down, UpSample


class UNet(linen.Module):
  """UNet."""

  out_channels: int
  kernel_size: int = 3
  stride: int = 1
  padding: str = "SAME"
  with_bn: bool = True
  bn_config: dict = field(
      default_factory=lambda: {
          "momentum": 0.99,
          "epsilon": 1e-4,
          "use_scale": True,
          "use_bias": True,
      }
  )

  @linen.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.numpy.ndarray:
    """Call."""
    x1 = DoubleConv(  # type: ignore
        out_channels=self.out_channels,
        kernel_size=self.kernel_size,
        stride=self.stride,
        padding=self.padding,
        with_bn=self.with_bn,
        bn_config=self.bn_config,
    )(x, train=train)
    x1 = linen.Dropout(rate=0.5)(x1, deterministic=not train)
    x2 = Down(out_channels=self.out_channels * 2)(x1, train=train)
    x3 = Down(out_channels=self.out_channels * 4)(x2, train=train)
    x4 = Down(out_channels=self.out_channels * 8)(x3, train=train)
    x5 = Down(out_channels=self.out_channels * 16)(x4, train=train)

    x = UpSample(out_channels=self.out_channels * 8)(x5, x4, train=train)
    x = UpSample(out_channels=self.out_channels * 4)(x, x3, train=train)
    x = UpSample(out_channels=self.out_channels * 2)(x, x2, train=train)
    x = UpSample(out_channels=self.out_channels)(x, x1, train=train)

    return linen.Conv(
        features=1,
        kernel_size=(self.kernel_size, self.kernel_size),
        strides=self.stride,
        padding=self.padding,
    )(x)


def test_unet() -> None:
  """Test unet."""
  x = jnp.ones((1, 1000, 1000, 3))
  rngs = jax.random.key(42)
  model = UNet(64)
  variables = model.init(rngs, x, train=False)
  y, mvars = model.apply(
      variables, x, train=True, rngs={"dropout": rngs}, mutable=["batch_stats"]
  )
  jax.tree_util.tree_map_with_path(
      lambda kp, mv: print(kp, mv.shape), flatten_dict(mvars, sep="/")
  )
  assert y.squeeze().shape == (1000, 1000)
  print(model.tabulate(rngs, x, train=True, depth=1))


if __name__ == "__main__":
  test_unet()
