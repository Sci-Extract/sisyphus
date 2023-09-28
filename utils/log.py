import logging

def log(log_file:str):
    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    file_handler.setLevel(logging.WARNING)
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[file_handler, stream_handler]
    )
    return logger