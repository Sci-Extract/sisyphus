"""evaluate model output based on ground truth data."""
from typing import Optional, NamedTuple

import dspy
from dspy.utils.parallelizer import ParallelExecutor
from pydantic import BaseModel

from .utils import dump_json
from ..utils.tenacity_retry_utils import pydantic_validate_retry_wraps

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('debug_evaluation.log', encoding='utf-8', mode='w')
logger.addHandler(fh)


class Metric(BaseModel):
    TP: int = dspy.OutputField(
        desc="True Positives: Count of entities in the LLM's output that matches entities in the ground truth. Treat each dictionary as a whole."
    )
    FP: int = dspy.OutputField(
        desc="False Positives: Count of entities in the LLM's output that do not exist in the ground truth. Treat each dictionary as a whole."
    )
    FN: int = dspy.OutputField(
        desc="False Negatives: Count of entities in the ground truth that are missing from the LLM's output. Treat each dictionary as a whole."
    )


class Evaluation(dspy.Signature):
    """Evaluate the performance of a large language model based information extraction task.
    Giving the evavluation metrics and errors if exits."""

    ground_truths: str = dspy.InputField(desc='ground truths, each dictinoary represents a single entity')
    llm_extracts: str = dspy.InputField(desc='extracted data from LLM model, each dictinoary represents a single entity')
    errors: str = dspy.OutputField(desc='verbal description of errors if exits. Describe missing or extra data points or detail errors of extraction entity, be succinct')
    metrics: Metric = dspy.OutputField(desc='evaluation metrics')


class Evaluate(dspy.Module):
    def __init__(self, temperature: float = 0.0):
        self.evaluator = dspy.ChainOfThought(Evaluation, temperature=temperature)
    
    def forward(self, ground_truths: str, llm_extracts: str, number_entities: int):
        prediction = self.evaluator(ground_truths=ground_truths, llm_extracts=llm_extracts)
        metric = prediction.metrics
        dspy.Suggest(
            result=(metric.TP + metric.FN == number_entities),
            msg="The sum of True Positives and False Negatives should be equal to the number of entities in the ground truth, check your answer again."
        )
        return prediction

def calculate_F1(prediction):
    metric = prediction.metrics
    precision = metric.TP / (metric.TP + metric.FP) if metric.TP != 0 else 0
    recall = metric.TP / (metric.TP + metric.FN) if metric.TP != 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    F1 = round(F1, 2)
    return F1

def scoring(predictions):
    tp_sum = sum(prediction.metrics.TP for prediction in predictions)
    fp_sum = sum(prediction.metrics.FP for prediction in predictions)
    fn_sum = sum(prediction.metrics.FN for prediction in predictions)
    F1 = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum) if tp_sum != 0 else 0
    F1 = round(F1, 2)
    return F1


class EvaluatedAttempt(NamedTuple):
    example: dspy.Example
    llm_extracts: Optional[list[BaseModel]]
    errors: str
    F1: float


def prioritized_fields_default(ground, predict):
    """customize this if the evaluation should prioritize certain fields"""
    return ground, predict

def exec_eval_parallel(program, last_key, examples, num_threads, prioritized_fields=prioritized_fields_default) -> tuple[list[EvaluatedAttempt], float]:
    executor = ParallelExecutor(num_threads, max_errors=0, disable_progress_bar=False)
    evaluated_attempts = []

    @pydantic_validate_retry_wraps
    def process_one(example):
        prediction = program(**example.inputs())

        ground_truths_p, llm_extracts_p = prioritized_fields(example[last_key], getattr(prediction, last_key))
        number_entities = len(ground_truths_p)
        llm_extracts_repr = dump_json(llm_extracts_p)
        ground_truths_repr = dump_json(ground_truths_p)
        evaluation = evaluator(ground_truths=ground_truths_repr, llm_extracts=llm_extracts_repr, number_entities=number_entities)
        F1 = calculate_F1(evaluation)

        example_copy = example.copy().with_inputs(*example.inputs().keys()) # copy the example and modify output
        example_copy[last_key] = ground_truths_p
        evaluated_attempts.append(
            EvaluatedAttempt(
                example=example_copy,
                llm_extracts=llm_extracts_p,
                errors=evaluation.errors,
                F1=F1
            )
        )
        return evaluation

    evaluations = executor.execute(process_one, examples)
    score = scoring(evaluations)
    for attempt in evaluated_attempts:
        text = dict(attempt.example.inputs())
        gt = attempt.example[last_key]
        pred = attempt.llm_extracts
        error = attempt.errors
        f1 = attempt.F1
        logger.debug('text: %s\nGround truth: %s\nPrediction: %s\nError: %s\nF1: %s\n', text, gt, pred, error, f1)
    return evaluated_attempts, score
  
evaluator = Evaluate().activate_assertions()
