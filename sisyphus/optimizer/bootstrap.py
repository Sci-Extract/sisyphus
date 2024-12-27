from functools import partial

import dspy


class BoostrapInstructions(dspy.Signature):
    """You are an instruction optimizer for large language models. Your task is to propose an instruction that will lead a language model to perform the task well. Don't be afraid to be creative."""
    task_description: str = dspy.InputField(desc='description of the task')
    proposed_instruction: str = dspy.OutputField(desc='The improved instruction for the language model')

bootstrapper_partial = partial(dspy.Predict, BoostrapInstructions)

def bootstrapper_agent(task_description: str, temperature, n):
    bootstrapper = bootstrapper_partial(temperature=temperature, n=n)
    return [c.proposed_instruction for c in bootstrapper(task_description=task_description).completions]
