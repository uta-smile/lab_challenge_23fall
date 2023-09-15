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
from typing import Callable
from .layers import DoubleConv, Down, UpSample
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
import haiku as hk
import optax


@dataclass
class UNet(hk.Module):
  """UNet."""

  out_channels: int
  kernel_size: int = 3
  stride: int = 1
  padding: str = "SAME"
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
    x1 = DoubleConv(  # type: ignore
        out_channels=self.out_channels,
        kernel_size=self.kernel_size,
        stride=self.stride,
        padding=self.padding,
        with_bn=self.with_bn,
        bn_config=self.bn_config,
    )(x, train=train)
    x2 = Down(out_channels=self.out_channels * 2)(x1, train=train)  # type: ignore
    x3 = Down(out_channels=self.out_channels * 4)(x2, train=train)  # type: ignore
    x4 = Down(out_channels=self.out_channels * 8)(x3, train=train)  # type: ignore
    x5 = Down(out_channels=self.out_channels * 16)(x4, train=train)  # type: ignore

    x = UpSample(out_channels=self.out_channels * 8)(x5, x4, train=train)  # type: ignore  # noqa: E501
    x = UpSample(out_channels=self.out_channels * 4)(x, x3, train=train)  # type: ignore
    x = UpSample(out_channels=self.out_channels * 2)(x, x2, train=train)  # type: ignore
    x = UpSample(out_channels=self.out_channels)(x, x1, train=train)  # type: ignore

    return hk.Conv2D(  # type: ignore
        output_channels=1,
        kernel_shape=self.kernel_size,
        stride=self.stride,
        padding=self.padding,
        with_bias=True,
    )(x)


def loss_fn(
    model: hk.TransformedWithState,
) -> Callable[
    [hk.Params, hk.State, jax.Array, jax.numpy.ndarray, jax.numpy.ndarray],
    jax.numpy.ndarray,
]:
  """Loss function."""

  apply = jax.jit(model.apply)

  def loss_fn(
      params: hk.Params,
      states: hk.State,
      rng: jax.Array,
      x: jax.numpy.ndarray,
      y: jax.numpy.ndarray,
  ) -> jax.numpy.ndarray:
    """Loss function."""
    y_pred, states = apply(params, states, rng, x)
    return optax.sigmoid_binary_cross_entropy(y_pred.squeeze(), y).mean(), states

  return loss_fn


def build_model() -> hk.TransformedWithState:
  """Build model."""

  def _unet(x: jax.Array, train: bool = True) -> jax.numpy.ndarray:
    return UNet(out_channels=64)(x, train=train)  # type: ignore

  return hk.transform_with_state(_unet)


unet: hk.TransformedWithState = build_model()


def test_unet() -> None:
  """Test unet."""
  x = jnp.ones((1, 1000, 1000, 3))
  rng = jax.random.PRNGKey(42)
  params, state = unet.init(rng, x)
  y, state = unet.apply(params, state, rng, x)
  assert y.shape == (1, 1000, 1000, 1)


if __name__ == "__main__":
  test_unet()
