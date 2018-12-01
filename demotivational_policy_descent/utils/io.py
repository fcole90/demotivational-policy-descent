import os

IO_FILE_PATH = os.path.realpath(__file__)

UTILS_PATH = os.path.dirname(IO_FILE_PATH)

MODELS_PATH= os.path.join(UTILS_PATH, os.pardir, "save_models")
