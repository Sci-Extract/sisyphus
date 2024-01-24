# {"compounds": [{"mof": "Ag-MOF-303", "uptake": {"temperature": {"value": 298.0, "unit": "K"}, "pressure": {"value": 0.2, "unit": "bar"}, "gas": "Xe", "result": {"value": 59.0, "unit": "cm3\u2009cm\u22123"}}}], "from_file": "ange.202015262"}

import pandas as pd
import json

# in_file = 'test_results_debug_app.jsonl'

def show_csv(in_file):
    raw_json = []
    with open(in_file, encoding='utf-8') as f:
        for line in f:
            raw_json.append(json.loads(line)[1])

    # modified_json = []
    # # mof temp pres gas val unit from
    # for json_ in raw_json:
    #     for compound in json_["content"]["compounds"]:
    #         uptake = compound["uptake"]
    #         collection = dict(mof=compound["mof"], temp=uptake["temperature"]["value"],
    #                           pres=uptake["pressure"]["value"], gas=uptake["gas"], val=uptake["result"]["value"], unit=uptake["result"]["unit"],
    #                           )
    #         modified_json.append(collection)

    modified_json = []
    # mof temp pres gas val unit from
    for json_ in raw_json:
        if json_ == "Failed":
            continue
        for compound in json_["content"]["compounds"]:
            for shg in compound["shg"]:
                collection = dict(
                    nlo=compound["name"],
                    shg_value=shg["value"],
                    shg_unit=shg["unit"],
                    dij=shg["dij"],
                    eg_value=compound["eg"]["value"],
                    eg_unit=compound["eg"]["unit"],
                    birefringence_value=compound["birefringence"]["value"],
                    birefringence_unit=compound["birefringence"]["unit"],
                    cutoff_value=compound["cutoff"]["value"],
                    cutoff_unit=compound["cutoff"]["unit"],
                    lidt_value=compound["lidt"]["value"],
                    lidt_unit=compound["lidt"]["unit"],
                )
                modified_json.append(collection)

    df = pd.DataFrame(modified_json)
    return df
    # df.dropna(subset=['birefringence_value'], inplace=True)
    # print(df)
    # df.to_excel("to_show_with_shg.xlsx")
# df = show_csv()
# df.dropna(subset=['birefringence_value'], inplace=True)
# df.to_excel("shg_drop.xlsx")
# modified_json = []
# d = {'compounds': [{'name': 'PbB5O7F3', 'shg': [{'value': 2.16, 'unit': 'pm/V', 'dij': 'd15'}, {'value': -0.94, 'unit': 'pm/V', 'dij': 'd24'}, {'value': -1.44, 'unit': 'pm/V', 'dij': 'd33'}], 'eg': {'value': None, 'unit': None}, 'birefringence': {'value': None, 'unit': None}, 'cutoff': {'value': None, 'unit': None}, 'lidt': {'value': None, 'unit': None}}, {'name': 'KDP', 'shg': [{'value': 0.39, 'unit': 'pm/V', 'dij': None}], 'eg': {'value': None, 'unit': None}, 'birefringence': {'value': None, 'unit': None}, 'cutoff': {'value': None, 'unit': None}, 'lidt': {'value': None, 'unit': None}}, {'name': 'CaB5O7F3', 'shg': [{'value': None, 'unit': None, 'dij': None}], 'eg': {'value': None, 'unit': None}, 'birefringence': {'value': None, 'unit': None}, 'cutoff': {'value': None, 'unit': None}, 'lidt': {'value': None, 'unit': None}}, {'name': 'SrB5O7F3', 'shg': [{'value': None, 'unit': None, 'dij': None}], 'eg': {'value': None, 'unit': None}, 'birefringence': {'value': None, 'unit': None}, 'cutoff': {'value': None, 'unit': None}, 'lidt': {'value': None, 'unit': None}}]}
# for compound in d["compounds"]:
#     for shg in compound["shg"]:
#         collection = dict(
#             nlo=compound["name"],
#             shg_value=shg["value"],
#             shg_unit=shg["unit"],
#             dij=shg["dij"],
#             eg_value=compound["eg"]["value"],
#             eg_unit=compound["eg"]["unit"],
#             birefringence_value=compound["birefringence"]["value"],
#             birefringence_unit=compound["birefringence"]["unit"],
#             cutoff_value=compound["cutoff"]["value"],
#             cutoff_unit=compound["cutoff"]["unit"],
#             lidt_value=compound["lidt"]["value"],
#             lidt_unit=compound["lidt"]["unit"],
#         )
#         modified_json.append(collection)
# df = pd.DataFrame(modified_json)
# df.to_csv("sample.csv", index=False)