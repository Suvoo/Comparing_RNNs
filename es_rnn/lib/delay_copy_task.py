import numpy as np
from torch.utils import data


class DelayCopyData(data.Dataset):
    """Delay-copy task data generator."""

    def __init__(self, seed=3000, length=10, width=3, initial_delay=0, initial_delay_fixed_length=True, delay=100,
                 delay_fixed_length=True, batch_size=10, binary_encoding=True, blank_symbol=True):
        """Initializes the data generator.
        """

        self.rng = np.random.default_rng(seed)
        self.length = length
        self.width = width
        self.initial_delay = initial_delay
        self.initial_delay_fixed_length = initial_delay_fixed_length
        self.delay = delay
        self.delay_fixed_length = delay_fixed_length
        self.batch_size = int(batch_size)
        self.binary_encoding = binary_encoding
        self.blank_symbol = blank_symbol

    # def next(self):
    def __getitem__(self, index):
        """
        """
        if self.binary_encoding:
            if self.width <= 8:
                tmp_width = self.width
            else:
                tmp_width = 8

            str_ = np.unpackbits(self.rng.choice(range(1, 2 ** tmp_width), self.length).astype(np.uint8).reshape(-1, 1),
                                 axis=1)[:, -tmp_width:]
            if self.width > 8:
                str_2 = np.unpackbits(
                    self.rng.choice(range(0, 2 ** (self.width - tmp_width)), self.length).astype(np.uint8).reshape(-1, 1), axis=1)[:, -(self.width - tmp_width):]
                str_ = np.concatenate((str_, str_2), axis=1)         
        else: # One-hot encoding
            symbols = self.rng.integers(0, self.width, size=(1, self.length))
            str_ = np.zeros((self.length, self.width))
            str_[np.arange(self.length), symbols] = 1

        if self.initial_delay_fixed_length:
            delay_before_str = self.initial_delay
        else:
            delay_before_str = self.rng.choice(range(1, self.initial_delay + 1))

        if self.delay_fixed_length:
            delay_after_str = self.delay
        else:
            delay_after_str = self.rng.choice(range(1, self.delay + 1))

        string_input = np.concatenate((np.zeros((delay_before_str, self.width)),
                                       str_,
                                       np.zeros((delay_after_str + self.length, self.width)),
                                       np.zeros((self.initial_delay - delay_before_str + self.delay - delay_after_str, self.width))),
                                      axis=0)

        recall_symbol = np.zeros((self.initial_delay + 2 * self.length + self.delay, 1))
        recall_symbol[delay_before_str + self.length + delay_after_str, 0] = 1

        if self.blank_symbol:
            blank_symbol = np.concatenate((np.ones((delay_before_str, 1)),
                                        np.zeros((self.length, 1)),
                                        np.ones((delay_after_str, 1)),
                                        np.zeros((1, 1)), # Recal symbol
                                        np.ones((self.length - 1 + self.initial_delay - delay_before_str + \
                                                    self.delay - delay_after_str, 1))), # Target
                                        axis=0)

            input_ = np.concatenate((string_input, recall_symbol, blank_symbol), axis=1).astype(np.float32)
        else:
            input_ = np.concatenate((string_input, recall_symbol), axis=1).astype(np.float32)
            
        target = str_.astype(np.float32)
        mask = np.zeros((self.initial_delay + self.delay + 2 * self.length, ))
        ind_start = delay_before_str + self.length + delay_after_str
        mask[ind_start : ind_start + self.length] = 1

        assert np.sum(mask) == self.length
        mask = np.repeat(mask[:, np.newaxis], self.width, axis=1)

        episode = {}
        episode['x'] = input_
        episode['y'] = target
        episode['mask'] = mask.astype(np.bool)
        # episode['datalen'] = data_len

        # print(input_.shape, target.shape, mask.shape)

        return episode

    def __len__(self): # denotes the total number of samples
        return 100000 * self.batch_size
