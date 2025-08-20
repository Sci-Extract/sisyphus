from pydantic import BaseModel


class Material(BaseModel):
    name: str


class PaperResult(BaseModel):
    properties: list[BaseModel] 
    synthesis: list[dict]

print(PaperResult(
    properties=[Material(name="Alice"), Material(name="Bob")],
    synthesis=[}"]
))
