import json

import dspy
from pydantic import BaseModel

def convert_to_dict(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(i) for i in obj]
    elif isinstance(obj, dspy.Example):
        d = dict(obj)
        return convert_to_dict(d)
    else:
        return obj

def dump_json(obj):
    obj_d = convert_to_dict(obj)
    return json.dumps(obj_d, indent=2, ensure_ascii=False)
