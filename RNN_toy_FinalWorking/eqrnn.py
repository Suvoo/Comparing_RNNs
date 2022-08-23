import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx


class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=lkey)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), carry

        out, stacked = lax.scan(f, hidden, input)

        logits = jax.vmap(self.linear)(stacked)
        return logits
