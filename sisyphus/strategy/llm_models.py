import dspy

from sisyphus.strategy.prompt_general import CATEGORIZE_DSPY


class SynCategorization(dspy.Signature):
    """system prompt"""
    text: str = dspy.InputField(description="The experimental section of a high entropy alloys (HEAs) research paper.")
    output: bool = dspy.OutputField()
SynCategorization.__doc__ = CATEGORIZE_DSPY
categorize_agent = dspy.ChainOfThought(SynCategorization)