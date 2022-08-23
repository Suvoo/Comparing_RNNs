import numpy as np

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
from jax import grad, jit, vmap

import optax
import equinox as eqx

import torch

from dataJax2 import *
from eqrnn import *


learning_rate = 0.001
steps = 10000
hidden_size= 50

# model_key = key
model_key = jrandom.split(key)

insize = width + 1
outsize = width

if blank_symbol:
    insize = width + 2

model = RNN(in_size= insize, out_size = outsize, hidden_size=hidden_size, key=model_key)

@eqx.filter_value_and_grad
def compute_loss(model, x, y,pred_y):
    
    # print(type(pred_y),type(y))
    criterion = optax.sigmoid_binary_cross_entropy(pred_y,y)
    loss = jnp.mean(criterion)
    return loss

    # Trains with respect to binary cross-entropy
    # return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))
    
    # Cross Entropy loss
    # return -jnp.sum(y*jnp.log(pred_y))
    # return -jnp.mean(y*jnp.log(pred_y))

@eqx.filter_jit
def make_step(model, x, y,pred_y,opt_state):

    loss, grads = compute_loss(model, x, y, pred_y)
    # print('grads is',grads)
    print('-----------------------------------------------------')
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state, pred_y

optim = optax.adam(learning_rate)
opt_state = optim.init(model)


print('key before gen',key)
old_key = key

for step in range(steps):
    key = jrandom.split(key, num=batch_size)
    vf = vmap(custom2)
    inputs,targets,masks = vf(key)
    key = key[batch_size - 1]

    rnn_out = jax.vmap(model)(inputs)
    # print('predy is ',rnn_out.shape)
    msk = rnn_out[masks]
    pred_y = msk.reshape(batch_size,length,width)
    # print(msk.shape)
    loss, model, opt_state,logits = make_step(model, inputs, targets, pred_y, opt_state)
    loss = loss.item()

    print(f"step={step}, loss={loss}")

    print(logits.shape, targets.shape)
    # print('inputs',inputs)
    # print('targets',targets)
    # print('=================================================')
    # print('logits',logits)

    np_array = np.asarray(logits)
    relevant_outputs = torch.from_numpy(np_array)
    # print(type(relevant_outputs))

    actual_output = torch.where(torch.sigmoid(relevant_outputs) < 0.5,
                    torch.zeros_like(relevant_outputs),
                    torch.ones_like(relevant_outputs))

    # print('actual',actual_output)

    targets = np.asarray(targets)
    targets = torch.from_numpy(targets)
    bitwise_success_rate = ((actual_output == targets).float().sum()) / (torch.numel(targets))

    ans = bitwise_success_rate/batch_size * 100

    print(bitwise_success_rate)
    print(ans)


# print('old key is',old_key)
# print('new key is',key)
# key = old_key
# print('now new key is',key)
