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
filename : train.py
project  : lab_exp_2023f
license  : GPL-3.0+

Train UNet.
"""
import pickle
from pathlib import Path

from data.data import Loader
from data.input import read_images

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from models.models import UNet, build_model, loss_fn
from tqdm import tqdm
from utils.utils import save

import tyro
from config import Conf


def train(conf: Conf) -> None:
  """Train model."""
  data = read_images(conf.data.base / "mnseg")
  loader = Loader(data, conf)

  model = build_model()
  rngs = hk.PRNGSequence(42)
  params, states = model.init(next(rngs), jnp.ones((1, 1000, 1000, 3)))

  optim = optax.lion(0.0001)
  opt_states = optim.init(params)
  lossfn = jax.jit(loss_fn(model))
  gradfn = jax.jit(jax.value_and_grad(lossfn, has_aux=True))

  with open("chks/34.pkl", "rb") as f:
    pkl = pickle.load(f)
    params, states, opt_states = pkl["params"], pkl["states"], pkl["opt_states"]

  for epoch in range(35, 100):
    losses = []
    for i, (x, y) in enumerate(loader):
      (loss, states), grads = gradfn(params, states, next(rngs), x, y)
      conf.log(f"Epoch {epoch} Batch {i} Loss {loss.item()}")
      updates, opt_states = optim.update(grads, opt_states, params)
      params = optax.apply_updates(params, updates)
      losses.append(loss.item())
    with open(f"chks/{epoch}.pkl", "wb") as f:
      pickle.dump({"params": params, "states": states, "opt_states": opt_states}, f)
    conf.log(f"Epoch {epoch} Loss {sum(losses) / len(losses)}")


def predict(conf: Conf) -> None:
  """Predict."""
  data = read_images(conf.data.base / "mnseg")
  loader = Loader(data, conf)
  model = build_model()
  rngs = hk.PRNGSequence(42)
  with open("chks/72.pkl", "rb") as f:
    pkl = pickle.load(f)
    params, states = pkl["params"], pkl["states"]

  apply = jax.jit(model.apply)
  ys = []
  ys_true = []
  dices = []
  for x, y in tqdm(Loader(data, conf)):
    y_, _ = apply(params, states, next(rngs), x)
    y_ = jnp.where(jax.nn.sigmoid(y_.squeeze()) > 0.5, 1, 0)
    ys.append(y_)
    ys_true.append(y.squeeze())
    dice = 2 * jnp.sum(y_ * y.squeeze()) / (jnp.sum(y_) + jnp.sum(y.squeeze()))
    dices.append(dice)
  save(ys, Path("ys.csv"))
  save(ys_true, Path("ys_true.csv"))
  conf.log(f"Dice {sum(dices) / len(dices)}")

if __name__ == "__main__":
  # train(Conf())
  predict(Conf())
