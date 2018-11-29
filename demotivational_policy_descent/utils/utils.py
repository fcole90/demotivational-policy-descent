import torch
import gym
from functools import reduce
import operator
import logging
import os


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


def prod(iterable):
    return reduce(operator.mul, list(iterable), 1)


def load_logger(filename, level=None):
    if level is None:
        level = logging.INFO
    path = os.path.join(os.pardir, "logs", filename + ".log")
    logging.basicConfig(level=level,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=path,
                        filemode='a+')
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    logging.info("Logger set up. Saving to '{}'".format(path))
