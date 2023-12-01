"""
Utility:
logger || counter || xlsx to csv || get cosine similarity
"""
import logging
import os
import time
import json

import pandas as pd
from termcolor import cprint


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
        
class ErrorRequestsTracker:
    def __init__(self):
        self.task_id = []
    
    def get_errors_id(self, result_jsonl_file_path: str):
        self.task_id = [] # initialize it for every times
        with open(result_jsonl_file_path, encoding="utf-8") as file:
            for line in file:
                line_json = json.loads(line)
                if line_json[1] == "Failed": # indicate this request was failed
                    self.task_id.append(line_json[2]["task_id"])
        if not bool(len(self.task_id)): # no error
            cprint("No errors, head to next step\n")
            return False
        return True
        
    def construct_redo_jsonl(self, jsonl_file_path: str):
        """make sure the jsonl file contains only singleton task.
        """
        redo_tasks = []
        with open(jsonl_file_path, encoding='utf-8') as file:
            for line in file:
                line_json = json.loads(line)
                if line_json["metadata"]["task_id"] in self.task_id:
                    redo_tasks.append(line_json)
        redo_file_path = jsonl_file_path.replace('.jsonl', '_redo.jsonl')
        with open(redo_file_path, 'w', encoding='utf-8') as file:
            for task in redo_tasks:
                file.write(json.dumps(task, ensure_ascii=False) + '\n')
        return redo_file_path
    
    def merge_back(self, redo_path, primal_path):
        responses = []
        if os.path.exists(redo_path):
            with open(redo_path, encoding='utf-8') as f:
                for line in f:
                    line_json = json.loads(line)
                    if line_json[1] != "Failed":
                        responses.append(line_json)
            if responses:
                self.write_to_file(responses, primal_path)

            open(redo_path, 'w', encoding='utf-8').close() # remove in case of duplicate collection.
    
    def write_to_file(self, contents, file_path, write_mode='a'):
        with open(file_path, write_mode, encoding='utf-8') as f:
            for content in contents:
                f.write(json.dumps(content, ensure_ascii=False) + '\n')

    def remove_fails(self, jsonl_file_path):
        temp_list = []
        with open(jsonl_file_path, encoding='utf-8') as f:
            for line in f:
                temp_list.append(json.loads(line))

        without_fails = [line_json for line_json in temp_list if line_json[1] != "Failed"]
        self.write_to_file(without_fails, jsonl_file_path, write_mode='w')
        

class Elapsed:
    def __init__(self, func):
        self.elapsed : float = 0
        self.func = func

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        ret = self.func(*args, **kwargs)
        end = time.perf_counter()
        self.elapsed += end - start
        cprint(f"Process finished, Runtime: {end - start:.2f}s\n", "red", attrs=["bold"])
        return ret
    