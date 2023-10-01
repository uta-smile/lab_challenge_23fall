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
date     : Sep  1, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : config.py
project  : lab_exp_2023f
license  : GPL-3.0+

SMILE LAB Tranining Project
"""

from dataclasses import dataclass, field
from typing import Any
import tyro
from pathlib import Path
from rich import console


@dataclass
class Data:
  """Dataset config."""

  raw: Path = field(default=Path("./raw"))
  base: Path = field(default=Path("./data"))

@dataclass
class Params:
  """Hyper Parameters."""

  lr: float = 1e-4
  batch_size: int = 2


@dataclass
class Conf:
  """Configuration."""

  data: Data = field(default_factory=Data)
  params: Params = field(default_factory=Params)
  seed: int = 7
  epoch: int = 200

  train: bool = True
  infer: bool = True
  predict: bool = True
  have_mask: bool = False

  def __post_init__(self) -> None:
    """Post initialize."""
    self.console = console.Console()

  def log(self, msg: Any) -> None:  # noqa: ANN401
    """Log MSG."""
    self.console.log(msg)

  def print(self, msg: Any) -> None:  # noqa: ANN401, A003
    """Print MSG."""
    self.console.print(msg)


if __name__ == "__main__":
  conf = tyro.cli(Conf)
  conf.print(conf)
