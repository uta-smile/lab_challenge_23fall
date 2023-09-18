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
from tqdm import tqdm
import jax


def rle(arr: jax.Array) -> str:
  """Run length encoding."""
  arr1d = arr.flatten()

  run_length = 0
  start_pixel = 0
  rles = []
  for i, elem in enumerate(tqdm(arr1d)):
    if elem not in (0, 1):
      raise ValueError("Only 0 and 1 are supported.")
    if elem:
      if run_length == 0:
        start_pixel = i + 1
      run_length += 1
    elif run_length:
      rles.append(f"{start_pixel} {run_length}")
      run_length = 0
  if run_length:
    rles.append(f"{start_pixel} {run_length}")
  return " ".join(rles)


def save(arrs: list[jax.Array], path: Path) -> None:
  """Save the array."""
  with path.open("w") as f:
    f.write("img,pixels\n")
    for i, arr in enumerate(arrs):
      f.write(f"{i},{rle(arr)}\n")
