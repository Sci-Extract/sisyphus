import dspy
from dspy.datasets import DataLoader
from typing import Literal, Optional
from pydantic import BaseModel, Field

from sisyphus.optimizer.intention import guess_intention
from sisyphus.optimizer.bootstrap import bootstrapper_partial
from sisyphus.utils.helper_functions import load_from_curated_examples
from sisyphus.optimizer.optimizer import Compiler


lm = dspy.LM('openai/gpt-4o-mini', cache=False)
dspy.configure(lm=lm)

class Target(BaseModel):
    target_formula: str = Field(description='make sure it is a valid chemical formula')
    amount_var: dict[str, list[float]] = Field(description='the amount variable in the formula, e.g. AxBC, {x: [1, 2]}')
    element_var: dict[str, list[str]] = Field(description='the element variable in the formula')

class Reaction(BaseModel):
    precursors: list[str] = Field(description='ensure it is a valid chemical formula')
    additives: list[str]
    target: Target
    reaction_type: str = Field(description='choose from solid-state, flux, hydrothermal, sol-gel, co-precipitation and other')

class QA(dspy.Signature):
    """extract chmemical reactions consitituent from the text."""
    text: str = dspy.InputField(desc='a piece of text which may contains chemical reactions')
    reactions: Optional[list[Reaction]] = dspy.OutputField(desc='the reactions extracted from the text, return null if no reaction found')

class ExtractReactionWithType(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought(signature=QA, temperature=0)

    def forward(self, text):
        prediction = self.predictor(text=text)
        return prediction
    
def prioritize_fields(ground, predict):
    ground_dict_p = [
        {
            'precursors': g.precursors,
            'target': g.target.target_formula
        }
        for g in ground
    ]

    if predict is None:
        predict_dict_p = None
        return ground_dict_p, predict_dict_p

    predict_dict_p = [
        {
            'precursors': p.precursors,
            'target': p.target.target_formula
        }
        for p in predict
    ]
    return ground_dict_p, predict_dict_p


dataset = load_from_curated_examples('curated_examples.json', ('text', 'reactions'), ('text',), Reaction)
splits = DataLoader().train_test_split(dataset, train_size=0.56, random_state=22)
train, dev = splits['train'], splits['test']
# prompt_model = dspy.LM('openai/gpt-4o', cache=False)
prompt_model = None
# compiler = Compiler(prioritize_field_func=prioritize_fields, prompt_model=prompt_model)
# compiled = compiler.compile(program=ExtractReactionWithType(), train_set=train, val_set=dev)

from sisyphus.optimizer.evaluator import exec_eval_parallel

_, score = exec_eval_parallel(ExtractReactionWithType(), 'reactions', dev, 10, prioritize_fields)
print(score)
