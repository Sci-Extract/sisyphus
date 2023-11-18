"""
Utility:
logger || counter || xlsx to csv || get cosine similarity
"""
import logging
import os
import pandas as pd


log_dir_path = os.path.join(os.getcwd(), "log")
def log(log_file_name="log.txt", logging_level=10):
    log_file = os.path.join(log_dir_path, log_file_name)
    
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

# count the # of function call during execute
class Counter:
    def __init__(self, func):
        self.counts = 0
        self.func = func
    
    def __call__(self, *args, **kwargs):
        ret = self.func(*args, **kwargs)
        self.counts += 1
        return ret, self.counts
    

class X2C:
    def __init__(self,file_name):
        # just the name without suffix
        self.file_name = file_name

    def convert(self):
        file_read = self.file_name + ".xlsx"
        df = pd.read_excel(file_read)
        file_out = self.file_name + ".csv"
        df.to_csv(file_out, index=False) 
        