import numpy as np
import torch
from lib.delay_copy_task import DelayCopyData
from torch.utils.data import DataLoader

seed=3000
length=10
width=3
initial_delay=0
initial_delay_fixed_length=True
delay=100
delay_fixed_length=True
batch_size=10
binary_encoding=True
blank_symbol=True




dataset = DelayCopyData(seed, length, width, initial_delay, initial_delay_fixed_length, delay,
                        delay_fixed_length, batch_size, binary_encoding, blank_symbol)

print(dataset)
loader = DataLoader(dataset, batch_size=batch_size)

data_iter = iter(loader)

data = next(data_iter)

# print(data[0])

inputs = (np.asarray(data['x']))
targets = (np.asarray(data['y']))
masks = (np.asarray(data['mask']))

print(inputs.shape)
print(type(inputs))
# print(inputs)

print(targets.shape)
print(type(targets))

print(masks.shape)
print(type(masks))