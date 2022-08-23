import functools
import math
from typing import Tuple, TypeVar
import warnings

import haiku as hk
import jax
import jax.numpy as jnp
import optax as optix
import numpy as np
# import pandas as pd
# import plotnine as gg


from dataJax2 import *

T = TypeVar('T')
Pair = Tuple[T, T]

model_key = key


def unroll_net(seqs: jnp.ndarray):
  """Unrolls an LSTM over seqs, mapping each output to a scalar."""
  # seqs is [T, B, F].
  core = hk.LSTM(32)
  batch_size = seqs.shape[1]
  outs, state = hk.dynamic_unroll(core, seqs, core.initial_state(batch_size))
  return hk.BatchApply(hk.Linear(1))(outs), state

model = hk.transform(unroll_net)
opt = optix.adam(1e-3)

@jax.jit
def loss(params, x, y):
    pred, _ = model.apply(params, None, x)
    # return jnp.mean(jnp.square(pred - y))

    criterion = optix.sigmoid_binary_cross_entropy(pred,y)
    loss = jnp.mean(criterion)
    return loss

@jax.jit
def update(step, params, opt_state, x, y):
    l, grads = jax.value_and_grad(loss)(params, x, y)
    grads, opt_state = opt.update(grads, opt_state)
    params = optix.apply_updates(params, grads)
    return l, params, opt_state

# Initialize state.
key = jrandom.split(key, num=10)
vf = vmap(custom2)
inputs,targets,masks = vf(key)
key = key[9]
# sample_x, _ = next(train_ds)
params = model.init(model_key, inputs)
opt_state = opt.init(params)

for step in range(300):
    key = jrandom.split(key, num=10)
    vf = vmap(custom2)
    inputs,targets,masks = vf(key)
    key = key[9]
    # if step % 100 == 0:
    # x, y = next(train_ds)
    train_loss, params, opt_state = update(step, params, opt_state, inputs, targets)
    print("Step {}: train loss {}".format(step, train_loss))
