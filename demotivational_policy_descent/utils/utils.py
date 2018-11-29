import torch
import gym
from functools import reduce
import operator
import logging
import os

# Path to this file
UTILS_FILE_PATH = os.path.realpath(__file__)

# Path to main app
SOURCE_ROOT_FILE_PATH = os.path.join(os.path.dirname(UTILS_FILE_PATH), os.pardir)

# Logs path
LOGS_PATH = os.path.join(SOURCE_ROOT_FILE_PATH, "logs")


def softmax_sample(ps):
    dist = torch.distributions.Categorical(ps)
    return dist.sample().item()


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def get_space_dim(space):
    t = type(space)
    if t is gym.spaces.Discrete:
        return space.n
    elif t is gym.spaces.Box:
        return space.shape[0]
    else:
        raise TypeError("Unknown space type:", t)

# Same as sum but for product
def prod(iterable):
    return reduce(operator.mul, list(iterable), 1)


def load_logger(filename, level=None):
    if level is None:
        level = logging.INFO
    path = os.path.join(LOGS_PATH, filename + ".log")
    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=path,
                        filemode='a+')
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    logging.info("Logger set up. Saving to '{}'".format(path))

def alert_on_cuda():
    cuda_art = """
                     _    _ _                   _      
             _ ___ _(_)__| (_)__ _   __ _  _ __| |__ _  (R)
            | ' \ V / / _` | / _` | / _| || / _` / _` |
 powered by |_||_\_/|_\__,_|_\__,_| \__|\_,_\__,_\__,_|
    """
    logging.info(cuda_art)


