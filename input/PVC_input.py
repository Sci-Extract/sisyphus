query = "Exploration of photovoltaic cell (PVC) (sometimes called solar cells) and associated parameters. The PVC is composed of an electron donor and an electron acceptor (e.g., PTB7-Th:DICTF-C10, where the left side acts as the donor and the right side as the acceptor). Key parameters for PVC encompass power conversion efficiency (PCE), short-circuit voltage (Jsc), fill factor (FF), and open circuit voltage (Voc)."

system_message = "You are reading a piece of text from chemistry articles about photovoltaic cells and you are required to response based on the context provided by the user."

prompt_cls = \
"""Given a text quoted by triple backticks, judge whether the text contains the desired information. Return a JSON object with the following criteria:

a. Check if the text includes at least a pair of phtovoltaic cell (PVC) e.g., PDTBTBO:ITIC.
b. Verify if the text includes at least one PVC parameter, such as power conversion efficiency (PCE), short-circuit voltage (Jsc), fill factor (FF), and open circuit voltage (Voc).
c. Confirm that the text contains at least one numerical value (e.g. PCE=6.93 %, Jsc=15.3 mA/cm2, FF=0.5, Voc= 0.940 V) corresponding to a PVC parameters.

Sequentially examine each criterion. If any criterion is not met, return False for that criterion. Output a JSON complying with the schema:

{
  "a": true/false,
  "b": true/false,
  "c": true/false
}

"""

prompt_sum = \
"""Given a text quoted by triple backticks containing information about photovoltaic cell (PVC).
Extract only PVC and corresponding parameters with numerical values in JSON format. 
The PVC should be separated to two parts, which is donor and acceptor (e.g., PTB7-Th:DICTF-C10 should be separated to PTB7-Th and DICTF-C10, where left is donor and right is acceptor) and the parameters to be extracted include power conversion efficiency (PCE), short-circuit voltage (Jsc), fill factor (FF), and open circuit voltage (Voc).
The JSON format should adhere to the following structure:

{
  "PVCs":[
    {
      "Donor": "<name>",
      "Acceptor": "<name>",
      "PCE": {"value": "<value>", "unit": "%"},
      "Jsc": {"value": <value>, "unit": "mA/cm2"},
      "FF": {<value>, "unit": null},
      "Voc": {<value>, "unit": "V"}
    },
    // Additional PVC follow the same structure if multiple nlo compounds are found
  ]  
}

Filled with null if "value" for any parameter is not found.
Exclude PVC that have null values for all properties.

"""