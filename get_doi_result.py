file = 'heas_bulk.json'
import json
import sys

with open(file, 'r') as f:
    data = json.load(f)

doi = sys.argv[1]
doi_results = [entry for entry in data if doi == entry.get('doi', '')]

with open('doi_result.json', 'w') as f:
    json.dump(doi_results, f, indent=2, ensure_ascii=False)