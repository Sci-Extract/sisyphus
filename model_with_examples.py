import re
from typing import Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field


#### Configurable section #####
# example of filter, remember that customizable functions must conform with this schema
def regex_filter(doc: Document) -> bool:
    shg_unit = re.compile(r'\b((pm\s*/\s*V)|(pm\s*V\s*-1))\b', re.I)
    eg_unit = re.compile(r'\beV\b')
    wavelength_unit = re.compile(r'\bnm\b')
    return bool(shg_unit.search(doc.page_content) or eg_unit.search(doc.page_content) or wavelength_unit.search(doc.page_content))

def regex_filter_shg(doc:Document) -> bool:
    shg_unit = re.compile(r'\b((pm\s*/\s*V)|(pm\s*V\s*-1))\b', re.I)
    shg_unit_standard = re.compile(r'(KDP)|(AgGaS2)')
    return bool(shg_unit.search(doc.page_content) or shg_unit_standard.search(doc.page_content))

# example of pydantic model
class ExtractNloProperties(BaseModel):
    """Extract properties of NLO(nonlinear optical) materials from text, some of them may not given in text, leave it to None"""

    nlo_name: str = Field(..., description='the chemical formula or name of nlo material')
    shg: Optional[float] = Field(
        None,
        description='the shg value of nlo material, precise value or times of some compound, e.g., 0.8 pm/V, 3 × KDP'
    )
    shg_unit: Optional[str] = Field(None, description='the unit of the shg value, pm/V or standard referenced material like KDP, AgGaS2')
    eg: Optional[float] = Field(None, description='the band gap or eg value of nlo material, e.g., 6.2 eV')
    eg_unit: Optional[str] = Field(None, description='the unit of the eg value, normally eV')
    cutoff: Optional[float] = Field(None, description='cutoff edge of nlo material, e.g., 225 nm')
    cutoff_unit: Optional[str] = Field(None, description='unit for cutoff, usually nm')
    birefringence: Optional[float] = Field(None, description='Δn, the birefringence value of nlo material, 0.55@1064 nm')
    birefringence_position: Optional[str] = Field(None, description='correpond position of birefringence, at which wavelength, e.g., 1064 nm')

class ExtractSHG(BaseModel):
    """Extract NLO property SHG(second harmonic generation) response/coefficient"""
    nlo_name: str = Field(..., description='the chemical formula or name of nlo material')
    shg: float = Field(
        ...,
        description='the SHG response value of nlo material, represented as precise value or times of some compound, e.g., 0.8 pm/V, 3 × KDP'
    )
    shg_unit: str = Field(..., description='the unit of the shg value, pm/V or standard referenced material like KDP, AgGaS2')

# NOTE: I recommand you to provide examples to get result match to provided.
tool_examples = [
    (
        'Example: After much effort, we obtained a new Pb-containing fluorooxoborate PbB5O7F3 with a strong SHG response (approximately 6 × KDP), a large birefringence (cal. 0.12@1064 nm), and a short UV cutoff edge (∼225 nm).',
        [
            ExtractNloProperties(
                nlo_name='PbB5O7F3',
                shg=6,
                shg_unit='KDP',
                cutoff=225,
                cutoff_unit='nm',
                birefringence=0.12,
                birefringence_position='1064 nm'
            ),
        ],
    ),
    (
        'Example: In this work, we developed a new DUV birefringent crystal LiBO2 based on [BO2]∞ infinite chains in the Li-B-O system, '
        'which simultaneously achieves the shortest UV cutoff edge (164 nm) and the largest birefringence (≥0.168 at 266 nm) among all the reported borate-based DUV birefringent materials.',
        [
            ExtractNloProperties(
                nlo_name='LiBO2',
                cutoff=164,
                cutoff_unit='nm',
                birefringence=0.168,
                birefringence_position='266 nm'
            )
        ],
    ),
    (
        'Example: Therefore, the effect of anion framework on the birefringence of SIO (cal. 0.093 at 1064 nm), SIOF (cal. 0.203 at 532 nm), and BIOF (cal. 0.092 at 1064 nm) crystals is vitally important (Figures S4–S6). '
        'Also, the calculated birefringence of SIOF is larger than those of the most reported fluoroiodates, such as RbIO2F2 (cal. 0.058 at 1064 nm) and CsIO2F2 (cal. 0.046 at 1064 nm).',
        [
            ExtractNloProperties(
                nlo_name='SIO',
                birefringence=0.093,
                birefringence_position='1064 nm',
            ),
            ExtractNloProperties(
                nlo_name = 'SIOF',
                birefringence=0.203,
                birefringence_position='532 nm'
            ),
            ExtractNloProperties(
                nlo_name='BIOF',
                birefringence='0.092',
                birefringence_position='1064 nm'
            ),
            ExtractNloProperties(
                nlo_name='RbIO2F2',
                birefringence=0.058,
                birefringence_position='1064 nm'
            ),
            ExtractNloProperties(
                nlo_name='CsIO2F2',
                birefringence=0.046,
                birefringence_position='1064 nm'
            )
        ]
    ),
]
tool_examples_single = [
    (
        'Example: After much effort, we obtained a new Pb-containing fluorooxoborate PbB5O7F3 with a strong SHG response (approximately 6 × KDP), a large birefringence (cal. 0.12@1064 nm), and a short UV cutoff edge (∼225 nm).',
        [
            ExtractSHG(
                nlo_name='PbB5O7F3',
                shg=6,
                shg_unit='KDP'
            ),
        ],
        
    ),
    (
        """Example: BBF exhibits a theoretically larger Eg (cal. ~8.88 eV), SHG effect (cal. d12 ~1.6×KDP) and Δn (cal. ~0.09) than KBBF, so that its shortest PM SHG output λPM (cal.) can reach 149 nm. Its deff at 177.3 nm is greater than KBBF, which meets the theoretical standard of DUV NLO crystals. ABF has a superior SHG effect (~3×KDP), Δn (~0.1) and λPM (~158 nm) than KBBF. Its effective SHG coefficient deff is twice that of KBBF, despite its Eg (~8 eV) is smaller than KBBF."""
        ,[
            ExtractSHG(
                nlo_name='BBF',
                shg=1.6,
                shg_unit='KDP'
            ),
            ExtractSHG(
                nlo_name='ABF',
                shg=3,
                shg_unit='KDP'
            )
        ]
    )
]
#### End #####
