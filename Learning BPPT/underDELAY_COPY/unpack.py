import numpy as np

seed=3000
length=10
width=3
initial_delay=0
initial_delay_fixed_length=True
delay=100
delay_fixed_length=True
batch_size=10
binary_encoding=True
blank_symbol=False

tmp_width = width

seed = 3000
rng = np.random.default_rng(seed)

str_ = np.unpackbits(rng.choice(range(1, 2 ** tmp_width), length).astype(np.uint8).reshape(-1, 1),
                                 axis=1)[:, -tmp_width:]
print(str_)

tmp_width = 8
str_2 = np.unpackbits(rng.choice(range(0, 2 ** -(width - tmp_width)), length).astype(np.uint8).reshape(-1, 1), axis=1)[:, -(width - tmp_width):]
print(str_2)
str_ = np.concatenate((str_, str_2), axis=1)
print(str_)



# print(str_)
# rng.choice gives arre b/w specified values
# a = [1,2,3,] but
# but reshape(-1,1) gives 2D array of [[],[],..]
# axis=1 gives 2d , and then we limit the lenght to width
# a = rng.choice(range(1, 2 ** tmp_width),length).astype(np.uint8).reshape(-1, 1)
# print('a is', a)
# b = np.unpackbits(a,axis=1)
# print('b is',b)
# a = np.unpackbits(rng.choice(range(1, 2 ** tmp_width), length).astype(np.uint8).

# ---------------------------------------------------------------------------------------------
# print(np.zeros((initial_delay - delay_before_str + delay - delay_after_str, width))) 0-0+100-100,3
# print(np.zeros((delay_after_str + length, width))) #100+10,3
# print(np.zeros((delay_before_str, width)),str_)

# print(str_.shape)
# print(np.zeros((delay_after_str + length, width)).shape)
# print('string_input is',string_input.shape)
# print(string_input)


# print(np.newaxis) https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
# https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
# print(mask[:, np.newaxis])
# print(np.repeat(mask[:, np.newaxis], width, axis=1))