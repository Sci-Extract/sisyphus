import json
from sisyphus.utils.helper_functions import get_create_resultdb


db = 'heas_bulk'
resultdb = get_create_resultdb(db)

with open(f'{db}.json', 'w') as f:
    data = resultdb.load_as_json(
        '',
        '',
        '',
        True
    )
    data = data[1:]
    json.dump(data, f, indent=2, ensure_ascii=False)