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

    file_handler.setLevel(logging.WARNING)
    stream_handler.setLevel(logging.DEBUG)

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s (%(filename)s) [%(levelname)s]: %(message)s",
        handlers=[file_handler, stream_handler]
    )
    return logger
