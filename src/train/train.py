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
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from functools import partial
from pathlib import Path

from data import build

import jax
import jax.numpy as jnp
import optax
from flax.core.scope import Collection
from flax.linen import Module
from flax.training import train_state

from models.models import UNet
from tqdm import tqdm
from utils.utils import save

import tyro
from config import Conf


class TrainState(train_state.TrainState):
  """Train State."""

  batch_stats: Collection
  loss: jax.Array


@partial(jax.jit, static_argnames=("train",))
def loss_fn(state: TrainState,
            x: jax.Array,
            y: jax.Array,
            rngs: jax.Array,
            train: bool = True) -> TrainState:
  """Loss function."""

  @partial(jax.value_and_grad, has_aux=True)
  def grad_fn(params: Collection) -> tuple[jax.Array, Collection]:
    """Calculate loss."""
    y_pred, updates = state.apply_fn(
        {
            "params": params,
            "batch_stats": state.batch_stats
        },
        x,
        train=train,
        rngs={"dropout": rngs},
        mutable="batch_stats",
    )
    return optax.sigmoid_binary_cross_entropy(y_pred.squeeze(), y).mean(), updates

  (loss, updates), grads = grad_fn(state.params)
  return state.apply_gradients(grads=grads, loss=loss, **updates)


def create_train_state(
    rngs: jax.Array,
    model: Module,
    shape: Sequence[int],
    hp: Collection,
) -> TrainState:
  """Create train state."""
  variables = model.init(rngs, jnp.ones((1, *shape)), train=False)
  tx = optax.lion(hp["lr"])
  return TrainState.create(
      apply_fn=model.apply,
      params=variables["params"],
      tx=tx,
      batch_stats=variables.get("batch_stats", {}),
      loss=jnp.zeros(()),
  )


def rng_gen(rngs: jax.Array) -> Iterator[jax.Array]:
  """Generate next rng key."""
  while True:
    rngs, subrng = jax.random.split(rngs)
    yield subrng


def train(conf: Conf) -> None:
  """Train model."""
  # Build model
  conf.log("Building model...")
  model = UNet(64)
  rngs = rng_gen(jax.random.key(conf.seed))
  print(
      model.tabulate(
          jax.random.key(conf.seed),
          jnp.ones((1, 1000, 1000, 3)),
          train=False,
          depth=1,
      ))

  # Init model
  conf.log("Building data and init params...")
  state = create_train_state(next(rngs), model, (1000, 1000, 3), asdict(conf.params))
  data = build(conf)
  conf.log("Building done...")

  conf.log("Start training...")
  dices = []
  for e in range(conf.epoch):
    losses = []
    for tri in tqdm(data.tr):
      # Get image embeddings
      im, mask = data.tr_embed.embed(tri)

      # forward and backward and get new params and states
      state = loss_fn(state, im, mask, next(rngs), train=True)
      losses.append(state.loss)

    ave_loss = jnp.asarray(losses).mean()
    conf.log(f"Epoch {e}, Loss {ave_loss}")

    # Evaluate
    dices.append(infer(conf, epoch=e, state=state))

    # Save model checkpoint
    with (Path("ckpts") / f"model-e{e:03d}.pkl").open("wb") as f:
      pickle.dump(
          {
              "params": state.params,
              "batch_stats": state.batch_stats,
              "opt_state": state.opt_state,
              "step": state.step,
              "loss": state.loss,
          },
          f,
      )
      d = jnp.stack(dices)
      conf.log(f"Max Dice at {jnp.argmax(d)} is {d.max()}")


def infer(conf: Conf, epoch: int = 0, state: TrainState | None = None) -> jax.Array:
  """Infer model."""
  conf.log("Start inference...")
  if state is None:
    model = UNet(64)
    state = create_train_state(
        jax.random.key(conf.seed), model, (1000, 1000, 3), asdict(conf.params))
    conf.log(f"Loading epoch {epoch}...")
    with (Path("ckpts") / f"model-e{epoch:03d}.pkl").open("rb") as f:
      pstate = pickle.load(f)  # noqa: S301
      state = state.replace(**pstate)

  data = build(conf, have_mask=conf.have_mask)
  dices = []
  for tei in tqdm(data.te):
    im, mask = data.te_embed.embed(tei)
    pred = state.apply_fn({
        "params": state.params,
        "batch_stats": state.batch_stats
    },
                          im,
                          train=False).squeeze()
    pred = jnp.where(jax.nn.sigmoid(pred) > 0.5, 1, 0)  # noqa: PLR2004
    dice = jnp.sum(pred * mask) * 2.0 / (jnp.sum(pred) + jnp.sum(mask))
    dices.append(dice)
  fdice = jnp.stack(dices).mean()
  conf.log(f"Epoch {epoch} Dice {fdice}")
  return fdice


def predict_kaggle(conf: Conf, epoch: int = 0, state: TrainState | None = None) -> None:
  """Infer model."""
  conf.log("Start predict...")
  if not state:
    model = UNet(64)
    state = create_train_state(
        jax.random.key(conf.seed), model, (1000, 1000, 3), asdict(conf.params))
    with (Path("ckpts") / f"model-e{epoch:03d}.pkl").open("rb") as f:
      pstate = pickle.load(f)  # noqa: S301
      state = state.replace(**pstate)
  data = build(conf, have_mask=conf.have_mask)
  preds = []
  masks = []
  for tei in tqdm(data.te):
    im, mask = data.te_embed.embed(tei)
    pred = state.apply_fn({
        "params": state.params,
        "batch_stats": state.batch_stats
    },
                          im,
                          train=False).squeeze()
    pred = jnp.where(jax.nn.sigmoid(pred) > 0.5, 1, 0)  # noqa: PLR2004
    preds.append(pred)
    masks.append(mask)
  save(jnp.asarray(preds), Path("ys_pred.csv"))
  if conf.have_mask:
    save(jnp.asarray(masks), Path("ys.csv"))


def main() -> None:
  """Run main function."""
  conf = tyro.cli(Conf)
  conf.log("Config:")
  conf.log(asdict(conf))
  if conf.train:
    train(conf)
  if conf.infer:
    infer(conf, conf.epoch - 1)
  if conf.predict:
    predict_kaggle(conf, conf.epoch - 1)


if __name__ == "__main__":
  main()
