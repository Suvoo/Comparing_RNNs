import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import grad, jit, vmap

from torch.utils.data import DataLoader

from dataJax2 import *

from delay_copy_task import *

steps = 1

def nptostr(nos):
    nos = np.asarray(nos)
    xs = nos.tolist()
    xs = [int(x) for x in xs]
    # print(xs)
    s = ''.join(str(x) for x in xs)
    return s

def batch_to_ar(targets):
    arr = []
    for i in targets:
        for t in i:
            # print(t,type(t))
            s = nptostr(t)
            # print(s)
            arr.append(s)
    return arr

def arr_to_dec(targets):
    ans = []
    arr = batch_to_ar(targets)
    for i in arr:
        s = nptostr(i)
        dec_number= int(s, 2)
        ans.append(dec_number)
    return ans

def JaxDataCheck(key):

    for step in range(steps):
        key = jrandom.split(key, num=batch_size)
        vf = vmap(custom2)
        inputs,targets,masks = vf(key)

        print('input shape :',inputs.shape)
        # print(type(inputs))
        # print(inputs)

        print('target shape:',targets.shape)
        # print(type(targets))

        print('masks shape :',masks.shape)

        # print(targets)

        # ans = arr_to_dec(targets)

        # print(ans)
        # # batch_to_ar(targets)

        # # nos = targets[0][0]
        # # binary = nptostr(nos)
        # # # print(binary,type(binary))
        # # dec_number= int(binary, 2)
        # # print(binary,dec_number)

        # key = key[batch_size - 1]
        # print('--------------------------------')

JaxDataCheck(key)

# seed=3000
# length=1
# width= 5
# initial_delay=0
# initial_delay_fixed_length=True
# delay=2
# delay_fixed_length=True
# batch_size= 10
# binary_encoding=True
# blank_symbol=False

def PytorchDataCheck():
    
    dataset = DelayCopyData(seed, length, width, initial_delay, initial_delay_fixed_length, delay,
                            delay_fixed_length, batch_size, binary_encoding, blank_symbol)

    loader = DataLoader(dataset, batch_size=batch_size)

    data_iter = iter(loader)

    for i in range(steps):
        data = next(data_iter)

        inputs = (np.asarray(data['x']))
        targets = (np.asarray(data['y']))
        masks = (np.asarray(data['mask']))

        print(inputs.shape)
        # print(type(inputs))
        # print(inputs)

        print(targets.shape)
        # print(type(targets))

        print(masks.shape)
        # print(type(masks))

        # print(targets)
        # ans = arr_to_dec(targets)

        # print(ans)
        # print('--------------------------------')

# PytorchDataCheck()