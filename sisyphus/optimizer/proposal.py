import dspy


class Proposing(dspy.Signature):
    """You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding F1 scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.

    Follow provided error patterns and suggestions, your task is to propose a new instruction that will lead language model to perform the task even better. Don't be afraid to be creative.""" # dspy docstring with some modifications (dspy.COPRO)

    task_description: str = dspy.InputField(desc='description of the information extraction task')
    attempted_instructions: str = dspy.InputField(desc='instructions that have been tried with their corresponding F1 scores')
    patterns: str = dspy.InputField(desc='high-level trends in the errors')
    suggestions: str = dspy.InputField(desc='suggestions for new instructions')
    proposed_instructions: str = dspy.OutputField(desc='The improved instructions for the language model. Please avoid over-complicated instructions')


class Propose(dspy.Module):
    def __init__(self, temperature: float = 0.7, n: int = 5):
        self.proposer = dspy.ChainOfThought(Proposing, temperature=temperature, n=n)

    def forward(self, task_description, attempted_instructions: str, patterns, suggestions: str):
        prediction = self.proposer(task_description=task_description, attempted_instructions=attempted_instructions, patterns=patterns, suggestions=suggestions)
        return prediction

def propose_agent(task_description, attempted_instructions: str, patterns, suggestions: str, temperature=0.7, n=5):
    proposer = Propose(temperature=temperature, n=n)
    prediction = proposer(task_description=task_description, attempted_instructions=attempted_instructions, patterns=patterns, suggestions=suggestions)
    return [c.proposed_instructions for c in prediction.completions]
