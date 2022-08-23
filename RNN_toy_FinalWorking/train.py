import optax
import torch
import math
import equinox as eqx

from delay_copy_task_jax import generate_delay_copy_data_flat
from eqrnn import RNN
import jax.random as jrandom
import jax.numpy as jnp
import jax
import numpy as np
from jax import vmap, jit

seed = 3000
length = 2
width = 3
initial_delay = 0
initial_delay_fixed_length = True
delay = 2
delay_fixed_length = True
batch_size = 1000
binary_encoding = True
blank_symbol = True

learning_rate = 0.001
steps = 10000
hidden_size = 50

insize = width + 1
outsize = width

if blank_symbol:
    insize = width + 2


def main():
    data_loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 2)
    model = RNN(in_size=insize, out_size=outsize, hidden_size=hidden_size, key=model_key)
    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)

    @eqx.filter_jit
    def compute_prediction(model, x):
        rnn_out = jax.vmap(model)(x)
        return rnn_out

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y, m):
        rnn_out = compute_prediction(model, x)
        y_logits = rnn_out[m]
        y_logits = y_logits.reshape(batch_size, length, width)
        criterion = optax.sigmoid_binary_cross_entropy(y_logits, y).sum(axis=-1)
        return criterion.mean()

    @eqx.filter_jit
    def update_model(model, grads, opt_state):
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    # @eqx.filter_jit
    def make_step(model, x, y, m, opt_state):
        loss, grads = compute_loss(model, x, y, m)
        return loss, *update_model(model, grads, opt_state)

    keys = data_loader_key
    for step in range(steps):
        keys = jrandom.split(keys, num=(batch_size + 1))

        vf = vmap(
            lambda kk: generate_delay_copy_data_flat(kk, length, width, initial_delay, initial_delay_fixed_length,
                                                     delay, delay_fixed_length, binary_encoding, blank_symbol))
        inputs, targets, masks = vf(keys[:-1])
        keys = keys[-1]

        loss, model, opt_state = make_step(model, inputs, targets, masks, opt_state)
        loss = loss.item()

        rnn_out = compute_prediction(model, inputs)
        y_logits = rnn_out[masks]
        y_logits = y_logits.reshape(batch_size, length, width)

        bitwise_success_rate = jnp.sum((jax.nn.sigmoid(y_logits) > 0.5) == targets) / jnp.prod(
            np.array([*targets.shape]))

        if bitwise_success_rate == 1.:
            print(f"step={step}, loss={loss}, bitwise_success_rate={bitwise_success_rate}")
            print(f"bitwise_success_rate has reached {bitwise_success_rate}. Breaking.")
            break

        if step % 100 == 0:
            print(f"step={step}, loss={loss}, bitwise_success_rate={bitwise_success_rate}")
    def evol():
        

main()
