import torch
import gym
from functools import reduce
import operator
import logging
import os
import subprocess
import io
import demotivational_policy_descent.utils.io as utils_io

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
                                                 ____ _   _ ____    __    
              _ _ __   __ __ ____   _    ___    / ___| | | |  _ \  /  \   (R)
             | '_ \ \ / /| ||  _ \ | |  / _ \  | |   | | | | | | |/ /\ \  
             | | | \ V / | || |_| || | / /_\ \ | |___| |_| | |_| / ____ \ 
 powered by  |_| |_|\_/  |_||____/ |_|/_/   \_\ \____|\___/|____/_/    \_\ 

    """
    logging.info(cuda_art)

def get_commit_hash():
    hash_value = ""
    try:
        proc = subprocess.Popen("git rev-parse HEAD".split(" "), stdout=subprocess.PIPE)
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            hash_value += line
    except Exception as e:
        logging.warning("Could not check hash value: {}".format(e))
        return "unknown"

    return hash_value[:-1]

def save_tmp_safe(agent, filename):
    latest_save_name = "tmp_" + filename + "_latest"
    latest_save_name_old = "tmp_" + filename + "_old"

    latest_save_name_old_path = os.path.join(utils_io.MODELS_PATH, latest_save_name_old) + ".pt"

    try:
        # Remove old tmp save
        if os.path.exists(latest_save_name_old_path):
            os.remove(latest_save_name_old_path)

        latest_save_name_path = os.path.join(utils_io.MODELS_PATH, latest_save_name) + ".pt"

        # Rename latest tmp to be the old tmp save
        if os.path.exists(latest_save_name_path):
            os.rename(latest_save_name_path, latest_save_name_old_path)
        else:
            logging.warning(latest_save_name_path + " did not exist")

        # Save the agent
        agent.save_model(latest_save_name)

    except Exception as e:
        logging.warning("Could not save: {}".format(e))


