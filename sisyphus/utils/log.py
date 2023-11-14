import logging
import os


def log(log_file="log.txt", logging_level=10):

    # create a log file if not exist
    if not os.path.exists(log_file):    
        with open(log_file, "w"):
            pass

    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s (%(filename)s) [%(levelname)s]: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    file_handler.setLevel(logging.WARNING)
    stream_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging_level)
    return logger
