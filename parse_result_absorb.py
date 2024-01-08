from pydantic import BaseModel, create_model

class Description(BaseModel):
    value: float | str | None
    unit: str | None
  
class Uptake(BaseModel):
    temperature: Description
    pressure: Description
    gas: str | None
    result: Description

class Compound(BaseModel):
    mof: str
    uptake: Uptake
  
class Compounds(BaseModel):
    compounds: list[Compound]


infile = "partial_results.jsonl"

import json
js_ls = []
with open(infile, encoding='utf-8') as file:
    for line in file:
        js_ls.append(json.loads(line))

parsed_json = []

for json_ in js_ls:
    try:
        parsed_json.append(Compound.model_validate(json_).model_dump())
    except Exception as e:
        print(e)
        continue

with open("MOF_results.jsonl", 'w', encoding='utf-8') as f:
    for json_ in parsed_json:
        f.write(json.dumps(json_))
        f.write('\n')