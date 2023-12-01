system_message = "You are reading a piece of text from chemistry articles about nonlinear optical (nlo) materials and you are required to response based on the context provided by the user."
prompt_sum = \
"""Given a text quoted by triple backticks containing information about nonlinear optical (nlo) compounds.
Extract only the nlo compounds and their corresponding properties with numerical values in JSON format. 
The compound name should be correct chemical formala and the properties to be extracted include the second harmonic generation coefficient (dij), band gaps (Eg), birefringence, absorption edge (cutoff edge), or LIDT (laser damage threshold).
The JSON format should adhere to the following structure:

{
  "compounds":[
    {
      "compound_name": <name>,
      "shg": {"value": <value>, "unit": <unit>},
      "eg": {"value": <value>, "unit": <unit>},
      "birefringence": {<value>, "unit": <unit>},
      "lidt": {<value>, "unit": <unit>}
    },
    // Additional compounds follow the same structure if multiple nlo compounds are found
  ]  
}

Exclude compounds that have null values for all properties.
If a property value is given by the times of a standard material, set the unit to the standard material (e.g., "unit": "KDP").

"""
prompt_cls = \
"""Given a text quoted by triple backticks, judge whether the text contains the desired information. Return a JSON object with the following criteria:

a. Check if the text includes at least one chemical compound (e.g., KBBF, BaB2O4, abbreviation, or pronoun).
b. Verify if the text includes at least one nonlinear optical (nlo) materials property corresponding to a specific chemical compound, such as second harmonic generation coefficient (dij), band gaps (Eg), birefringence, absorption edge (cutoff edge), or LIDT.
c. Confirm that the text contains at least one numerical value (e.g. 4.5 eV, 0.45 pm V-1) corresponding to a nonlinear optical (nlo) materials property.

Sequentially examine each criterion. If any criterion is not met, return False for that criterion. Output a JSON complying with the schema:

{
  "a": true/false,
  "b": true/false,
  "c": true/false
}

"""

query = "Description of the properties of NLO materials, include the name of nlo material (e.g. KBBF, Na4B8O9F10), second harmonic generation SHG (e.g. 0.8 pm/V, 3 × KDP), band gaps Eg (e.g. 6.2 eV), birefringence, phase match, absorption edge, laser induced damage thersholds (LIDT). reports values unit such as (eV, pm/V, MW/cm2, nm), and the SHG value is sometimes given in multiples of KDP or AgGaS2."