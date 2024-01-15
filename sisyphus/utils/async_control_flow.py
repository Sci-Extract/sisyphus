"""
Usage:
Module defines a shema for async control flow

Schema:
- Bucket
    - used for control the rate for async tasks (GCRA implemmentation, refer to wiki)
    - a bucket with some water inside, will decreased (for some action like making requests) and increased (recovery with time goes by)
- Tracker
    - track the state of the tasks, include #start taks, #in progress tasks, #failed tasks. In progress task drop to 0 then main loop exit
- Main loop
    - Dispatch tasks and re-dispatch failed ones.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generator

from sisyphus.utils.utilities import log


class Bucket:
    def __init__(self, maximum_capacity: float, recovery_rate: float, init_time: float = time.time(), init_capacity: float = None):
        self.maximum_capacity = maximum_capacity
        self.recovery_rate = recovery_rate
        self.last_update_time = init_time
        self.current_time: float = None
        
        if init_capacity is not None:
            if init_capacity > maximum_capacity:
                raise ValueError("init_capacity cannot bigger than maximum_capacity")
            self.present_capacity = init_capacity
        else:
            self.present_capacity = maximum_capacity

    def update(self):
        self.current_time = time.time()
        time_elapsed = self.current_time - self.last_update_time
        self.present_capacity = min(self.present_capacity + time_elapsed * self.recovery_rate,
                                    self.maximum_capacity)
        self.last_update_time = self.current_time

    def consume(self, consumption: float):
        self.present_capacity -= consumption

    def has_capacity(self, consumption) -> bool:
        return True if (self.present_capacity >= consumption) else False

@dataclass
class Tracker:
    task_start_num: int = 0
    task_in_progress_num: int = 0
    task_failed: int = 0
    task_failed_ls: list = field(default_factory=list)


class AsyncControler(ABC):
    def __init__(self, error_savefile_path: str, max_redo_times: int = 3, logging_level: int = 10):
        self.error_savefile_path = error_savefile_path
        self.max_redo_times = max_redo_times
        self.logger = log(logging_level=logging_level)

    async def control_flow(self, input: Generator, bucket: Bucket, tracker: Tracker, most_concurrent_task_num: int):
        """Theory:
        Spawn rate: bucket recovery_rate
        Digest rate: the implementation running rate
        Workers: most_concurrent_task_num
        when adujust these parameters, You can think there are some workers(concurrent tasks) who are transfering(implementation, digest process /dynamic/) goods(the spawn).
        note the unit is per implement consumption size per second. (maybe it's hard to conceive)
        case 1: 1 worker, spawn rate < digest rate.
            the consume speed is faster than the produce speed, which means the worker is waiting for the goods to be spawned. the overall rate is controled by spawn rate
        case 2: 1 worker, spawn rate > digest rate.
            the opposite of case 1, the overall rate is controled by digest rate.
        case 3: multiple workers
            with multiple workers, the rate limit is min(#workers * digest rate, spawn rate) e.g., workers = 3, digest rate = 1, spawn rate = 4, then 3 * 1 < 4, the overall rate is 3.
        """
        redo_queue = asyncio.Queue()
        sema = asyncio.Semaphore(most_concurrent_task_num)
        redo_times_d = defaultdict(int)
        input_run_out = False
        next_request = None
        while True:
            if not next_request:
                if not input_run_out:
                    try:
                        next_ = next(input)
                        next_request = self.implement(next_, tracker=tracker, sema=sema, redo_queue=redo_queue)
                        tracker.task_start_num += 1
                        tracker.task_in_progress_num += 1
                    except StopIteration:
                        input_run_out = True
                elif not redo_queue.empty():
                    next_ = redo_queue.get_nowait()
                    redo_times = redo_times_d[next_] + 1
                    next_request = self.implement(next_, tracker=tracker, sema=sema, redo_queue=redo_queue, redo_times=redo_times)
                    tracker.task_start_num += 1
            
            bucket.update()

            if next_request:
                consumption = self.request_consumption(next_)
                if bucket.has_capacity(consumption):
                    
                    asyncio.create_task(next_request)
                    bucket.consume(consumption=consumption)
                    next_request = None

            if tracker.task_in_progress_num == 0:
                break

            await asyncio.sleep(0.001) # brief sleep so that coroutines can run
        
        if len(tracker.task_failed_ls):
            with open(self.error_savefile_path, 'w', encoding='utf-8') as file:
                file.write("Failed:\n")
                for item in tracker.task_failed_ls:
                    file.write(f"{str(item)}\n")

    
    @abstractmethod
    def request_consumption(self, request):
        pass
    
    async def implement(self, request, tracker: Tracker, sema: asyncio.Semaphore, redo_queue: asyncio.Queue, redo_times: int = None):
        async with sema:
            try:
                result = await self.carrier(request)
                self._save(result)
                tracker.task_in_progress_num -= 1

            except Exception as e:
                self.logger.warning(f"{e}")
                tracker.task_failed += 1
                if redo_times < self.max_redo_times:
                    redo_queue.put_nowait(request)
                else:
                    tracker.task_failed_ls.append(request)

    @abstractmethod
    async def carrier(self, request, *args, **kwargs):
        """The task's core action, for example, requests.get, visit remote sql database etc."""
        pass

    @abstractmethod
    def _save(self, result):
        pass
