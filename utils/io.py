import os

IO_FILE_PATH = os.path.realpath(__file__)

MODELS_PATH = os.path.dirname(IO_FILE_PATH)

APP_ROOT_PATH = os.path.join(os.pardir, MODELS_PATH)
