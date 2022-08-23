import random
import time
from datetime import datetime

def get_random_name(prefix='baseline'):
    datetime_suffix = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S")
    randnum = str(random.randint(1e3, 1e5))
    sim_name = f"{prefix}-{randnum}-{datetime_suffix}"
    return sim_name


datetime_suffix = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S")
print(datetime_suffix)

randnum = str(random.randint(1000, 100000))
print(randnum)

prefix = 102
sim_name = f"{prefix}-{randnum}-{datetime_suffix}"
print(sim_name)

