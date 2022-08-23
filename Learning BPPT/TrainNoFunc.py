import argparse
import os
import yaml
import numpy as np

from underDELAY_COPY.dataJax2 import *

from es_rnn.lib import Timer, get_random_name, RNNType
from es_rnn.lib.models import RNNStatefulWrapper, RNNReadoutWrapper

import optax  # https://github.com/deepmind/optax
import equinox as eqx

# import underDELAY_COPY.dataJax2

key = jrandom.split(key, num=10)
vf = vmap(custom2)
inputs,targets,masks = vf(key)
print(inputs)

key = key[9]

key = jrandom.split(key, num=10)
vf = vmap(custom2)
inputs,targets,masks = vf(key)
print('-------------------------------------------------------------------------------------')
print(inputs)

input_size = c.total_input_width
if c.rnn_type == RNNType.LSTM:
    rnn = eqx.nn.LSTMCell(input_size, c.n_units, batch_first=True)
elif c.rnn_type == RNNType.GRU:
    rnn = eqx.nn.GRUCell(input_size, c.n_units, batch_first=True)
else:
    raise RuntimeError("Unknown lstm type: %s" % c.rnn_type)
print("LSTM parameters: ", list(map(lambda x: x[0], rnn.named_parameters())))

model = RNNReadoutWrapper(rnn, output_size=c.target_width)

train_params = list(model.parameters())

if c.use_rmsprop:
    print('Using RMSprop.')
    optimizer = optax.rmsprop(lr=c.learning_rate, decay=0.9)
    opt_state = optimizer.init(train_params)
else:
    optimizer = optax.adam(lr=c.learning_rate)
    opt_state = optimizer.init(train_params)

if c.binary_encoding:
    loss_function = nn.BCEWithLogitsLoss()
else:
    loss_function = nn.CrossEntropyLoss()
