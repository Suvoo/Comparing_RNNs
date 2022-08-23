import random
import time
from datetime import datetime
from enum import Enum

class Timer:
    def __init__(self):
        self._startime = None
        self._endtime = None
        self.difftime = None

    def __enter__(self):
        self._startime = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._endtime = time.time()
        self.difftime = self._endtime - self._startime


class RNNType(Enum):
    LSTM = 'lstm'
    GRU = 'gru'
    INDYLSTM = 'indylstm'

class Optimizer(Enum):
    adam = 'adam'
    sgd = 'sgd'


def get_random_name(prefix='baseline'):
    datetime_suffix = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S")
    randnum = str(random.randint(1e3, 1e5))
    sim_name = f"{prefix}-{randnum}-{datetime_suffix}"
    return sim_name
