from pydantic import BaseModel, create_model

M = create_model('Model', __base__=BaseModel, name=(str, ...))
B = create_model('Model', __base__=BaseModel, name=(str, ...))
m = M(name="Alice")
b = B(name="Bob")
print(m.__repr__())
print(b.__repr__())