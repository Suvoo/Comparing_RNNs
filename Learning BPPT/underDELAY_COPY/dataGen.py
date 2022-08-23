import numpy as np

# __init__ : for data generation
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

rng = np.random.default_rng(seed) #random number generator : https://numpy.org/doc/stable/reference/random/generator.html
print(rng)

# __getitem__ : for next, it has some parameter index:
index = 3
if binary_encoding: # True
    if width <= 8:  # width = 3 
        tmp_width = width
    else:
        tmp_width = 8
print('tmp_width is', tmp_width)

str_ = np.unpackbits(rng.choice(range(1, 2 ** tmp_width), length).astype(np.uint8).reshape(-1, 1),
                                 axis=1)[:, -tmp_width:]
# print(str_)

if width > 8: # width = 3 
    str_2 = np.unpackbits(
        rng.choice(range(0, 2 ** (width - tmp_width)), length).astype(np.uint8).reshape(-1, 1), axis=1)[:, -(width - tmp_width):]
    str_ = np.concatenate((str_, str_2), axis=1)

else: # One-hot encoding
    symbols = rng.integers(0, width, size=(1, length))
    # print(symbols)
    str_ = np.zeros((length, width))
    # print(np.arange(length), symbols)
    str_[np.arange(length), symbols] = 1
    # print(str_)

if initial_delay_fixed_length: #True
    delay_before_str = initial_delay
else:
    delay_before_str = rng.choice(range(1, initial_delay + 1))

if delay_fixed_length: #True
    delay_after_str = delay
else:
    delay_after_str = rng.choice(range(1, delay + 1))

string_input = np.concatenate((np.zeros((delay_before_str, width)),
                                str_,
                                np.zeros((delay_after_str + length, width)),
                                np.zeros((initial_delay - delay_before_str + delay - delay_after_str, width))),
                                axis=0)
print('string shape',string_input.shape)

recall_symbol = np.zeros((initial_delay + 2 * length + delay, 1)) # 0 + 2 * 10 + 100,1
# print(recall_symbol.shape)
# print(delay_before_str + length + delay_after_str, 0)
recall_symbol[delay_before_str + length + delay_after_str, 0] = 1
print('recall shape',recall_symbol.shape)

if blank_symbol: #True
    blank_symbol = np.concatenate((np.ones((delay_before_str, 1)),
                                np.zeros((length, 1)),
                                np.ones((delay_after_str, 1)),
                                np.zeros((1, 1)), # Recal symbol
                                np.ones((length - 1 + initial_delay - delay_before_str + \
                                            delay - delay_after_str, 1))), # Target
                                axis=0)
    print('blank shape',blank_symbol.shape)

    input_ = np.concatenate((string_input, recall_symbol, blank_symbol), axis=1).astype(np.float32)
    print('input_shape is',input_.shape)
    print(type(input_))
else:
    input_ = np.concatenate((string_input, recall_symbol), axis=1).astype(np.float32)

target = str_.astype(np.float32)
# print(target)
mask = np.zeros((initial_delay + delay + 2 * length, )) # 0 + 100 + 2*10
# print('mask is',mask)
ind_start = delay_before_str + length + delay_after_str # 110
# print('ind_start is',ind_start)
mask[ind_start : ind_start + length] = 1
print('mask is',mask.shape)
# print('mask is',mask)

assert np.sum(mask) == length
# print(np.sum(mask),length)

mask = np.repeat(mask[:, np.newaxis], width, axis=1)
print('new mask is ',mask.shape)
# print(np.newaxis) https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
# https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
# print(mask[:, np.newaxis])
# print(np.repeat(mask[:, np.newaxis], width, axis=1))

episode = {}
episode['x'] = input_
episode['y'] = target
episode['mask'] = mask.astype(np.bool)

print(input_.shape, target.shape, mask.shape)

# print(episode)
        
# __len__
print(100000 * batch_size)

