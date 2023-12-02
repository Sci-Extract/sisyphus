import json


file = "data\\completion_sum_results.jsonl"


import json

def transform_json(json_str):
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print("Invalid JSON string:", json_str)
        return None

    # Transform the data into your desired format

    transformed_json_str = json.dumps(data, indent=2)
    return transformed_json_str

def write_to_file(filename, *json_strings):
    with open(filename, 'w') as file:
        for json_str in json_strings:
            transformed_json = transform_json(json_str)
            if transformed_json:
                file.write(transformed_json + '\n')  # Write each transformed JSON to a new line

# Example usage
json_strings_ls = []
with open(file, encoding="utf-8") as f:
    for line in f:
        line_json = json.loads(line)
        if line_json[1] != "Failed":
            json_strings_ls.append(line_json[1]["choices"][0]["message"]["content"])

write_to_file('TEST_out_numericalmaybe.jsonl', *json_strings_ls)
