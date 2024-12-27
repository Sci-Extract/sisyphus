import random

import dspy
from pydantic import BaseModel

from .evaluator import EvaluatedAttempt
from .utils import dump_json
from .agent_prompt import reflexion_prompt


class Reflection(dspy.Signature):
    task_description: str = dspy.InputField(desc='description of the information extraction task')
    attempts_with_evaluation: list[dict] = dspy.InputField(desc='program extraction attempts with evaluation feedback')
    extract_instruction: str = dspy.InputField(desc='program extraction instruction')
    patterns: str = dspy.OutputField(desc='high-level trends in the errors.')
    suggestions: str = dspy.OutputField(desc='clear, actionable suggestions for improvement.')

Reflection.__doc__ = reflexion_prompt

class Reflect(dspy.Module):
    def __init__(self, temperature: float = 0.7):
        self.reflector = dspy.ChainOfThought(Reflection, temperature=temperature)

    def forward(self, task_description, attempts_with_evaluation: list[dict], extract_instruction):
        prediction = self.reflector(task_description=task_description, attempts_with_evaluation=attempts_with_evaluation, extract_instruction=extract_instruction)
        return prediction

class PatternWithSuggestion(BaseModel):
    patterns: str
    suggestions: str


def reflexion(task_description, evaluates: list[EvaluatedAttempt], extract_instruction, use_passed_num: int = 3, seed=22):
    """idea based on 'Reflexion: Language Agents with Verbal Reinforcement Learning'"""
    passed = []
    failed = []
    for evaluate in evaluates:
        if evaluate.F1 == 1.0:
            passed.append(
                {
                    **evaluate.example,
                    'program_extraction': evaluate.llm_extracts,
                    'feedback': 'matched'
                }
            )
        else:
            failed.append(
                {
                    **evaluate.example,
                    'program_extraction': evaluate.llm_extracts,
                    'feedback': evaluate.errors
                }
            )
    rng = random.Random(seed)
    use_passed = rng.sample(passed, use_passed_num)
    prediction = reflector(
        task_description=task_description,
        extract_instruction=extract_instruction,
        attempts_with_evaluation=dump_json(use_passed + failed)
    )
    return PatternWithSuggestion(patterns=prediction.patterns, suggestions=prediction.suggestions)

reflector = Reflect()
