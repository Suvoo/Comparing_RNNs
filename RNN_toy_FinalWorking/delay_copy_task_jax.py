import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


def generate_delay_copy_data(key, length, width, initial_delay, initial_delay_fixed_length, delay,
                             delay_fixed_length, binary_encoding, blank_symbol):
    # __getitem__ : for next, it has some parameter index:
    if binary_encoding:  # True
        if width <= 8:  # width = 3 
            tmp_width = width
        else:
            tmp_width = 8
        # maybe split the key here everytime

        str_ = jnp.unpackbits(jrandom.choice(key, np.array(range(1, 2 ** tmp_width)), (length,))
                              .astype(np.uint8).reshape(-1, 1), axis=1)[:, -tmp_width:]

        if width > 8:
            str_2 = jnp.unpackbits(
                jrandom.choice(key, np.array(range(0, 2 ** (width - tmp_width))), (length,)).astype(np.uint8).reshape(
                    -1, 1), axis=1)[:, -(width - tmp_width):]
            str_ = jnp.concatenate((str_, str_2), axis=1)

    else:  # One-hot encoding
        # split here maybe
        jsymbols = jrandom.randint(key, (1, length,), 0, width)
        str_ = jnp.zeros((length, width))
        str_ = str_.at[jnp.arange(length), jsymbols].set(1)  # str_[jnp.arange(length),jsymbols] = 1

    if initial_delay_fixed_length:
        delay_before_str = initial_delay
    else:
        # delay_before_str = rng.choice(range(1, initial_delay + 1))
        delay_before_str = jrandom.choice(key, np.array(range(1, initial_delay + 1)))

    if delay_fixed_length:
        delay_after_str = delay
    else:
        # maybe split
        delay_after_str = jrandom.choice(key, np.array(range(1, delay + 1)))  # check it gives jax array

    string_input = jnp.concatenate((jnp.zeros((delay_before_str, width)),
                                    str_,
                                    jnp.zeros((delay_after_str + length, width)),
                                    jnp.zeros((initial_delay - delay_before_str + delay - delay_after_str, width))),
                                   axis=0)

    recall_symbol = jnp.zeros((initial_delay + 2 * length + delay, 1))
    recall_symbol = recall_symbol.at[delay_before_str + length + delay_after_str, 0].set(1)

    if blank_symbol:
        blank_symbol = jnp.concatenate((jnp.ones((delay_before_str, 1)),
                                        jnp.zeros((length, 1)),
                                        jnp.ones((delay_after_str, 1)),
                                        jnp.zeros((1, 1)),  # Recal symbol
                                        jnp.ones((length - 1 + initial_delay - delay_before_str + \
                                                  delay - delay_after_str, 1))),  # Target
                                       axis=0)

        input_ = jnp.concatenate((string_input, recall_symbol, blank_symbol), axis=1)

    else:
        input_ = jnp.concatenate((string_input, recall_symbol), axis=1)

    target = str_.astype(float)
    mask = jnp.zeros((initial_delay + delay + 2 * length,))
    ind_start = delay_before_str + length + delay_after_str
    mask = mask.at[ind_start: ind_start + length].set(1)

    assert jnp.sum(mask) == length
    mask = jnp.repeat(mask[:, jnp.newaxis], width, axis=1)

    episode = {}
    episode['x'] = input_
    episode['y'] = target
    episode['mask'] = mask.astype(bool)

    return episode


def generate_delay_copy_data_flat(key, length, width, initial_delay, initial_delay_fixed_length, delay,
                                  delay_fixed_length, binary_encoding, blank_symbol):
    e = generate_delay_copy_data(key, length, width, initial_delay, initial_delay_fixed_length, delay,
                                 delay_fixed_length, binary_encoding, blank_symbol)
    inputs, targets, masks = e['x'], e['y'], e['mask']
    return inputs, targets, masks


if __name__ == '__main__':
    seed = np.random.randint(1e5)
    length = 1
    width = 3
    initial_delay = 1
    initial_delay_fixed_length = True
    delay = 2
    delay_fixed_length = True
    batch_size = 100
    binary_encoding = True
    blank_symbol = True

    key = jrandom.PRNGKey(seed)

    inputs, targets, masks = generate_delay_copy_data_flat(key, length, width, initial_delay,
                                                           initial_delay_fixed_length, delay,
                                                           delay_fixed_length, binary_encoding, blank_symbol)
    print(inputs, targets, masks)
