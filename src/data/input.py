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
filename : input.py
project  : lab_exp_2023f
license  : GPL-3.0+

Read in images.
"""
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from PIL import Image


def read_image(path: Path, binary: bool = False) -> jax.Array:
  """Read image from PATH."""
  if binary:
    return jnp.asarray(Image.open(path).convert("1"), dtype="float32")
  return jnp.asarray(Image.open(path), dtype="float32")


def read_images(path: Path, binary: bool = False) -> Sequence[jax.Array]:
  """Read images from PATH."""
  paths = sorted(path.glob("*.png"))
  return tuple(map(partial(read_image, binary=binary), paths))
