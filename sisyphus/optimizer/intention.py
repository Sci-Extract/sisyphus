"""guessing the user's intention (usage) from input-output examples"""
import json
import random

import dspy
from pydantic import BaseModel

from .utils import dump_json


class Intention(dspy.Signature):
    """Given input-output examples for information extraction tasks, please describe the nature of the task."""
    input_output_pairs: str = dspy.InputField(desc='input-output examples for a specific information extraction tasks')
    description: str = dspy.OutputField(desc='a succinct description of the task, e.g. "extracting name entities from text"')


def guess_intention(examples, use_num: int = 3, seed: int = 22):
    rng = random.Random(seed)
    predictor = dspy.Predict(Intention)
    use_examples = rng.sample(examples, use_num)
    dumped = dump_json(use_examples)
    prediction = predictor(input_output_pairs=dumped)
    return prediction.description
