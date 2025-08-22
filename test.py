
from dotenv import load_dotenv
load_dotenv()

from ast import literal_eval
import json

from typing import Optional
from pydantic import BaseModel, Field, field_validator

from sisyphus.heas.synthesis import get_synthesis_prompt

class Processing(BaseModel):
    """Processing route for a material
    Return field steps as '[]' if you cannot find any"""
    steps: str = Field(description="""List of processing steps in chronological order, form them as json object. For example: [{"induction melting": {"temperature": "1500 K"}}, {"annealed": {"temperature": "800 K", "duration": "1h"}}]""")

    @field_validator('steps', mode='after')
    @classmethod
    def load(cls, value: str):
        try:
            value = literal_eval(value)
        except:
            value = json.loads(value)
        return value

class Strength(BaseModel):
    """Tensile/Compressive test results"""
    ys: Optional[str] = Field(description="Yield strength with unit")
    uts: Optional[str] = Field(description="Ultimate tensile/compressive strength with unit")
    strain: Optional[str] = Field(description="Fracture strain with unit")
    temperature: Optional[str] = Field(description="Test temperature with unit")
    strain_rate: Optional[str] = Field(description="Strain rate with unit")
    other_test_conditions: Optional[str] = Field(description="Other test environment conditions, like in salt, hydrogen charging. Do not include processing/synthesis conditions.")

class Record(BaseModel):
    """Strength and processing belong to one material"""
    material: str
    strength: Optional[Strength]
    processing: Optional[Processing]

class Records(BaseModel):
    records: Optional[list[Record]]

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import dspy
lm = dspy.LM('openai/gpt-4.1-mini', temperature=0.0)
dspy.configure(lm=lm)

model = ChatOpenAI(model='gpt-4.1-mini', temperature=0.0)
instruction = '''You are an expert materials scientist tasked with extracting material properties and processing information from scientific text.
### **General Guidelines:**  
- **Extract data for all materials** with reported mechanical properties, even if they are:  
  - Mentioned in **comparisons** with other samples.  
  - Referenced from a **previous study** but include numerical properties in the current text.  
  - Not the main focus of the paragraph.
- If multiple materials are described under different conditions, **each must be recorded separately** with its corresponding processing conditions.  
'''
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'Extract properties along with processing steps'),
        ('human', "{instruction}\nformat instruction:{format}\ntext:{text}")
    ]
)

text = """
Experimental:
(CoCrFeNi)95Ti1Nb1Al3 HEA was prepared in a vacuum induction furnace using high-purity ( > 99%) elements. This was followed by homogenization at 1473 K for 12 h. Approximately 1 mm-thick plates were cut from the ingot and rolled to 0.1 or 0.2 mm, according to the experimental requirements. 0.2 mm-thick tensile samples, as shown in Fig. 1, and 0.1 mm-thick samples for transmission electron microscopy (TEM) examination were punched out from the rolled plates. 10 mm-long and 0.2 mm-thick square samples were prepared for ion irradiation. To eliminate defects introduced during processing, all samples were solution-treated in a 10-5 Pa vacuum at 1273 K for 1 h and then air-cooled. The conditions for precipitation were 1023 K for 8 h under vacuum. Electropolishing (set to 15 V) was performed using a HClO4 (25%) and CH3COOH (75%) solution to remove the oxide film formed on the sample surface by annealing.

2.2 Tensile test
Tensile specimens with a thickness of 0.2 mm were tested at room temperature with a cross-head speed of 1 mm/min, which corresponds to a strain rate of 2×10-3/s. The engineering strain and total elongation were approximated by dividing the cross-head displacement by the initial gauge length of the specimen, which had been calibrated [15,16].

Tensile test:
Fig. 6 shows the nominal tensile stress-strain curves for the (CoCrFeNi)95Ti1Nb1Al3 HEA samples annealed at 1273 K for 1 h and further aged at 1023 K for 8 h. For comparison, the tensile results of the CoCrFeNi HEA and CoCrFeMnNi HEA (well-studied HEA), which were annealed at 1273 K for 1 h, are also shown. The shape (especially the thickness) of the tensile sample, the synthesis method of the materials (such as vacuum arc melting, vacuum induction melting, and mechanical alloying followed by spark plasma sintering), and the impurities in the alloy affect the mechanical properties of the materials. However, the yield stress (184 MPa) and tensile strength (523 MPa) of the CoCrFeNi HEA, and the yield stress (230 MPa) and tensile strength (760 MPa) of the CoCrFeMnNi HEA obtained in this study agreed with the reported results [28]. The yield stress (190 MPa) and tensile strength (580 MPa) of the (CoCrFeNi)95Ti1Nb1Al3 HEA annealed at 1273 K for 1 h were higher than those of the CoCrFeNi HEA, but lower than those of the CoCrFeMnNi HEA. This result indicates that the addition of Ti, Nb, and Al with a total composition ratio of 5% slightly increased the yield stress and tensile strength of the equiatomic composition ratio CoCrFeNi HEA. After aging (CoCrFeNi)95Ti1Nb1Al3 HEA at high temperatures, precipitates were formed. The elongation was reduced, but the yield stress and tensile strength increased significantly. The temperature and time of aging clearly affected the mechanical properties of the (CoCrFeNi)95Ti1Nb1Al3 HEA. Fig. 6 shows the stress-strain curve with the highest values of the yield stress and tensile strength of the aged (CoCrFeNi)95Ti1Nb1Al3 HEA. The aging conditions were a temperature of 1023 K and time of 8 h. The yield stress and tensile strength increased significantly from 190 to 450 MPa and 580-870 MPa, respectively, and the total elongation deceased from 50% to 40%. The precipitates improved the strength of the (CoCrFeNi)95Ti1Nb1Al3 HEA.
"""
syn_text = """(CoCrFeNi)95Ti1Nb1Al3 HEA was prepared in a vacuum induction furnace using high-purity ( > 99%) elements. This was followed by homogenization at 1473 K for 12 h. Approximately 1 mm-thick plates were cut from the ingot and rolled to 0.1 or 0.2 mm, according to the experimental requirements. 0.2 mm-thick tensile samples, as shown in Fig. 1, and 0.1 mm-thick samples for transmission electron microscopy (TEM) examination were punched out from the rolled plates. 10 mm-long and 0.2 mm-thick square samples were prepared for ion irradiation. To eliminate defects introduced during processing, all samples were solution-treated in a 10-5 Pa vacuum at 1273 K for 1 h and then air-cooled. The conditions for precipitation were 1023 K for 8 h under vacuum. Electropolishing (set to 15 V) was performed using a HClO4 (25%) and CH3COOH (75%) solution to remove the oxide film formed on the sample surface by annealing.
"""

format = get_synthesis_prompt(text=syn_text, lm=lm)
print(format)
chain = prompt | model.with_structured_output(Records, method='json_schema')

r = chain.invoke({'text': text, 'instruction': instruction, 'format': format}).records
print(r)
print('\n')

# print(r.model_dump())
    