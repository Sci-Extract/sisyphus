"""
Convert text chunks to formatted json file for text embedding /* contain metadata */
"""

import json


MODEL = "text-embedding-ada-002"

def render2json(text: list[str], identifier: str, filename: str):
    requests = text
    jobs = [{"model": MODEL, "input": request, "metadata": {"file_name": identifier}} for request in requests]
    
    with open(filename, "a", encoding="utf-8") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + '\n')
