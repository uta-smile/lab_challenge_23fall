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
from config import Conf
from pathlib import Path
from PIL import Image
from collections.abc import Sequence
import jax
import jax.numpy as jnp


def read_image(path: Path, gray: bool = False) -> jax.Array:
  """Read image from PATH."""
  if gray:
    image = jnp.asarray(Image.open(path).convert("L"))
    return jnp.where(image > 127, 1, 0).astype("float32")  # noqa: PLR2004
  return jnp.asarray(Image.open(path), dtype="float32")


def read_images(path: Path) -> Sequence[tuple[jax.Array, jax.Array]]:
  """Read images from PATH."""
  paths = sorted((path / "im").glob("*.png"))
  return tuple(
      map(
          lambda x: (
              read_image(x),
              read_image(x.parent.parent / "mask" / x.name, gray=True),
          ),
          paths,
      )
  )


# def read_test(path: Path) -> jax.Array:
#   """Read test images from PATH."""
#   return tuple(map(read_image, sorted((path / "test").glob("*.png"))))
