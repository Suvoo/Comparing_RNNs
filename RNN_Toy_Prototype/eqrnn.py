import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx


from dataJax2 import *

class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear
    # bias: jnp.ndarray

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        # Check Linear Layer here and ,maybe remove it. also see linear layer documentation
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=lkey)
        # self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), carry

        out,stacked = lax.scan(f, hidden, input)

        # sigmoid because we're performing binary classification
        # return jax.nn.sigmoid(self.linear(out) + self.bias)
        # print(type(stacked))
        # return stacked

        # logit = self.linear(stacked) + self.bias # lineatr
        logit = jax.vmap(self.linear)(stacked)
        # # logit = out + self.bias
        # # print(out.shape)
        return logit

        # return jax.nn.softmax(self.linear(out) + self.bias)


# model_key = jrandom.split(key)   
# hidden_size = 50
# insize = width + 1
# outsize = width

# if blank_symbol:
#     insize = width + 2
# model = RNN(in_size= insize, out_size = outsize, hidden_size=hidden_size, key=model_key)

# print(model)
