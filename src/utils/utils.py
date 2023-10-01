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
date     : Sep 11, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : utils.py
project  : lab_exp_2023f
license  : GPL-3.0+

Utils
"""
from pathlib import Path
import numpy as np
import jax


def rle(arr: jax.typing.ArrayLike) -> str:
  """Run length encoding."""
  arr1d = np.asarray(arr).flatten()
  arr1d = np.pad(arr1d, (1, 1), mode="constant", constant_values=0)
  runs = np.where(arr1d[1:] != arr1d[:-1])[0] + 1
  runs[1::2] -= runs[::2]
  return " ".join(map(str, runs))


def save(arrs: list[jax.Array], path: Path) -> None:
  """Save the array."""
  with path.open("w") as f:
    f.write("img,pixels\n")
    for i, arr in enumerate(arrs):
      f.write(f"{i},{rle(arr)}\n")
