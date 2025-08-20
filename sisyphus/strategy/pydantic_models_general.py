from typing import Optional
from pydantic import BaseModel, Field

# context method
class Material(BaseModel):
    composition: Optional[str] = Field(description="Chemical composition of the material, e.g., 'Mn0.2CoCrNi'")

class MaterialDescriptionBase(BaseModel):
    composition: Optional[str] = Field(description="Chemical composition of the material, e.g., 'Mn0.2CoCrNi'")
    description: Optional[str] = Field(description="Description of the material. Give the description based on the processing method, e.g., 'as-cast', 'annealed at 900C'. Do not contain test condition which describes the testing setup rather than the material itself, such as tested under 700C, under salted environment. If material is given with composition only, without any description, return None.")
    refered: bool = Field(description="Indicate whether the material data is cited from other publications for comparison purposes.")

class Processes(Material):
    """Processing route for a material"""
    processes: str = Field(description="List of processing steps in chronological order, for each as a python dictionary. For example: [{'induction melting': {'temperature': 1500}}, {'annealed': {'temperature': 800, 'duration': '1h'}}]")

