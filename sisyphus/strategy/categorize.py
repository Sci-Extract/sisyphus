import dspy

class SynCategorization(dspy.Signature):
    """system prompt"""
    text: str = dspy.InputField(description="experimental section")
    output: bool = dspy.OutputField(description="indicate whether all materials are distinctly identified")

def get_categorize_agent(prompt):
    """return user defined categorize agent"""
    SynCategorization.__doc__ = prompt
    return dspy.ChainOfThought(SynCategorization)
