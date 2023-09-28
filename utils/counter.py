# count the # of function call during execute
class Counter:
    def __init__(self, func):
        self.counts = 0
        self.func = func
    
    def __call__(self, *args, **kwargs):
        ret = self.func(*args, **kwargs)
        self.counts += 1
        return ret, self.counts

# count the # of code line execution times
class SnippetCounter:
    def __init__(self):
        self.counts = 0

    def increment(self):
        self.counts += 1

import time

# count the time of some process
class TimeCounter:
    def __init__(self):
        pass
    
    def start(self):
        self._start_time = time.perf_counter()
    
    @property
    def start_time(self):
        return self._start_time
    
    def end(self):
        self._end_time = time.perf_counter()
    
    @property
    def end_time(self):
        return self._end_time
