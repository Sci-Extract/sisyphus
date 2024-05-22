# -*- coding:utf-8 -*-
"""
@File    :   database.py
@Time    :   2024/05/11 17:25:13
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   Create base model for document and result, update result base functionality and document sql db operation e.g., get with source.
Note that parameter sql_model refers to model without setting table=True, while sql_table is the opposite.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional, Sequence, Callable

from langchain.pydantic_v1 import BaseModel, create_model
from langchain_core.documents import Document
from sqlmodel import SQLModel, Field, Session, select, JSON, Relationship, text, create_engine, col
from sqlalchemy.orm import registry


def extend_fields(sql_model: SQLModel, pydantic_model: BaseModel) -> SQLModel:
    """update sqlmodel (not sql table!) with user defined pydantic model"""

    def get_default(value):
        if not value.required:
            return value.default
        return ...

    field_defs = {
        key: (value.annotation, get_default(value))
        for key, value in pydantic_model.__fields__.items()
    }

    return create_model('ResultBase', __base__=sql_model, **field_defs)


def doc_getter(engine, sql_table: SQLModel):
    """get doc using its source, always return one doc"""

    def get_article(source):
        with Session(bind=engine) as session, session.begin():
            result = session.exec(text(f"SELECT page_content, meta from document WHERE json_extract(meta, '$.source') = '{source}'")).all() # TODO: find a solution using orm, which is more generalizable, ^_^ I hate sql.
            assert result, f'not find given {source}'
            documents = [Document(page_content=res[0], metadata=json.loads(res[1])) for res in result]
        return documents

    return get_article


def get_new_sql_base():
    """Get a sql base with empty registry, for details refer to https://github.com/tiangolo/sqlmodel/issues/264"""

    class NewBase(SQLModel, registry=registry()):
        pass

    return NewBase


class DocBase(SQLModel):
    """base class of doc"""

    id: Optional[int] = Field(default=None, primary_key=True)
    page_content: str
    meta: dict = Field(
        sa_type=JSON
    )   # do not use metadata which will shadow sqlmodel predefined object


# used for grafting
DOCBASE_DEF = {
    'id': (Optional[int], Field(default=None, primary_key=True)),
    'page_content': (str, ...),
    'meta': (dict, Field(sa_type=JSON)),
}


class ResultBase(SQLModel):
    """base model for result"""

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: Optional[int] = Field(default=None, foreign_key='document.id')

RESULT_DEF = {
    'id': (Optional[int], Field(default=None, primary_key=True)),
    'document_id': (Optional[int], Field(default=None, foreign_key='document.id'))
}

class DB(ABC):
    @abstractmethod
    def _define_sqltable(self):
        pass

    @abstractmethod
    def create_db(self):
        pass

    def check_source(self, obj: dict):
        return 'source' in obj

class DocDB(DB):
    """provide functionality for creating database, saving data, searching through database"""

    def __init__(self, engine):
        """
        init DocDB

        Args:
            engine: sql engine
        """
        self.engine = engine
        self.NewBase = get_new_sql_base()
        self.Document = self._define_sqltable()

    def _define_sqltable(self) -> SQLModel:
        NewDocBase = create_model(
            'NewDocBase', __base__=self.NewBase, **DOCBASE_DEF
        )

        class Doc(NewDocBase, table=True):
            __tablename__ = 'document'

        return Doc

    def create_db(self):
        """invocate `SQLModel.metadata.create_all`"""
        self.NewBase.metadata.create_all(self.engine)

    def get(self, source):
        """get article using correspond source name"""
        getter = doc_getter(self.engine, self.Document)
        return getter(source)

    def save_texts(self, texts: list[str], metadatas: list[dict[str]]):
        """batch saving text with metadata to docdb"""
        assert super().check_source(metadatas[0]), 'metadata must have source field'
        with Session(self.engine) as session, session.begin():
            records = [
                self.Document(page_content=page_content, meta=metadata)
                for page_content, metadata in zip(texts, metadatas)
            ]
            for record in records:
                session.add(record)


class ResultDB(DB):
    """provide functionality for creating database, saving data, searching through database"""

    def __init__(self, engine, result_pydantic: BaseModel):
        self.engine = engine
        self.result_pydantic = result_pydantic
        self.NewBase = get_new_sql_base()
        self.Document, self.Result = self._define_sqltable()

    def _define_sqltable(self) -> SQLModel:
        NewDocBase = create_model(
            'NewDocBase', __base__=self.NewBase, **DOCBASE_DEF
        )
        NewResultBase = create_model(
            'NewResultBase', __base__=self.NewBase, **RESULT_DEF
        )
        ExtendResultBase = extend_fields(NewResultBase, self.result_pydantic)
        
        class Doc(NewDocBase, table=True):
            __tablename__ = 'document'

            results: list['Result'] = Relationship(back_populates='document')
        
        class Result(ExtendResultBase, table=True):
            document: Doc = Relationship(back_populates='results')
        
        return Doc, Result
    
    def create_db(self):
        """invocate `SQLModel.metadata.create_all`"""
        self.NewBase.metadata.create_all(self.engine)

    def get(self, source):
        """get article using correspond source name"""
        getter = doc_getter(self.engine, self.Document)
        return getter(source)
    
    def save_result(self, text: str, metadata: dict[str, str], results: list[BaseModel]):
        """save text with extracted resutls"""
        assert super().check_source(metadata), 'metadata must have source field'
        assert isinstance(results[0], self.result_pydantic), f'model mismatch, expect {self.result_pydantic}'
        with Session(self.engine) as session, session.begin():
            document = self.Document(page_content=text, meta=metadata)
            for result in results:
                params = dict(result)
                result_sqlmodel = self.Result(**params)
                document.results.append(result_sqlmodel)
            session.add(document)


NewBase = get_new_sql_base()

class ExtractRecord(NewBase, table=True):
    __tablename__ = 'extract_record'
    id: Optional[int] = Field(None, primary_key=True)
    key: str = Field(..., index=True)
    namespace: str
    # namespace is to isolated different extracted process


class ExtractManager:
    """the primary goal was to skip those articles which have been extracted. Design inspiration is originated from langchain index method.
    sync version support presently.
    - key, the name of the file
    """

    def __init__(self, namespace, db_url):
        self.namespace = namespace
        self.db_url = db_url
        self.engine = create_engine(db_url)
        
    def create_schema(self):
        NewBase.metadata.create_all(bind=self.engine)
    
    def exists(self, keys: Sequence[str]) -> list[bool]:
        """return booleans to indicates extracted or not"""
        with Session(bind=self.engine) as session, session.begin():
            records = session.exec(select(ExtractRecord).where(col(ExtractRecord.key).in_(keys)).where(ExtractRecord.namespace == self.namespace)).all()
            found_keys = set(r.key for r in records)
        return  [key in found_keys for key in keys]
    
    def update(self, key):
        record = ExtractRecord(key=key, namespace=self.namespace)
        with Session(bind=self.engine) as session, session.begin():
            session.add(record)
    
    async def aupdate(self, key):
        await asyncio.to_thread(self.update, key)


def add_manager_callback(func: Callable, manager: ExtractManager):
    """wrap a callable to use extract manager, must be a coroutine func!"""
    @wraps(func)
    async def wrapper(key):
        r = await func(key)
        await manager.aupdate(key)
        return r
    return wrapper
