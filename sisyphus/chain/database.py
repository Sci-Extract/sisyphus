# -*- coding:utf-8 -*-
"""
@File    :   database.py
@Time    :   2024/05/11 17:25:13
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   includes base models for Document and Result sqlmodel function for converting pydantic definitions to sqlmodel
"""

from typing import Optional

from langchain.pydantic_v1 import BaseModel, create_model
from sqlmodel import SQLModel, Field


class DocBase(SQLModel):
    """base model for document"""
    id: Optional[int] = Field(default=None, primary_key=True)
    page_content: str
    source: str
    title: str


class ResultBase(SQLModel):
    """base model for result"""
    id: Optional[int] = Field(default=None, primary_key=True)
    doc_id: int = Field(..., foreign_key='document.id')


def update_resultbase(result_model: BaseModel) -> SQLModel:
    """update result sqlmodel with user defined pydantic model"""
    def get_default(value):
        if not value.required:
            return value.default
        return ...

    field_defs = {
        key: (value.annotation, get_default(value))
        for key, value in result_model.__fields__.items()
    }

    return create_model('ResultBase', __base__=ResultBase, **field_defs)
