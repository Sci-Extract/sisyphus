# -*- coding:utf-8 -*-
"""
@File    :   database.py
@Time    :   2024/05/11 17:25:13
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   Create base model for doc database and result database. Note that parameter sql_model refers to sqlmodel without setting table to True.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional, Sequence, Callable, cast, Type, get_args, get_origin, Union

from pydantic import BaseModel, create_model
from langchain_core.documents import Document
from sqlmodel import SQLModel, Field, Session, select, JSON, Relationship, text, create_engine, col
from sqlalchemy.orm import registry


def extend_fields(sql_model: SQLModel, pydantic_model: BaseModel) -> SQLModel:
    """update sqlmodel (not sql table!) with user defined pydantic model"""

    def get_default(value):
        if not value.is_required():
            return value.default
        return ...

    field_defs = {
        key: (value.annotation, get_default(value))
        for key, value in pydantic_model.model_fields.items()
    }

    return create_model('ResultBase', __base__=sql_model, **field_defs)


def doc_getter(engine, sql_table: SQLModel, with_abstract: bool = False):
    """get doc using its source, always return one doc"""

    def get_article(source):
        with Session(bind=engine) as session, session.begin():
            result = session.exec(text(f"SELECT page_content, meta from documents WHERE json_extract(meta, '$.source') = '{source}'")).all() # TODO: find a solution using orm, which is more generalizable, ^_^ I hate sql.
            assert result, f'not find given {source}'
            if not with_abstract:
                documents = [Document(page_content=res[0], metadata=json.loads(res[1])) for res in result]
            else:
                page_meta_pairs = [(res[0], json.loads(res[1])) for res in result]
                abstract = ''
                for pair in page_meta_pairs:
                    if pair[1]['sub_titles'] == 'Abstract':
                        abstract = pair[0]
                        break
                assert abstract, f'not find abstract for {source}, you may not have sub_titles field in your database or the abstract of {source} is absent' # for testing, comment this when in production
                for pair in page_meta_pairs:
                    pair[1].update(abstract=abstract)
                documents = [Document(page_content=res[0], metadata=res[1]) for res in page_meta_pairs]
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
    document_id: Optional[int] = Field(default=None, foreign_key='documents.id')

RESULT_DEF = {
    'id': (Optional[int], Field(default=None, primary_key=True)),
    'document_id': (Optional[int], Field(default=None, foreign_key='documents.id'))
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
            __tablename__ = 'documents'

        return Doc

    def create_db(self):
        """invocate `SQLModel.metadata.create_all`"""
        self.NewBase.metadata.create_all(self.engine)

    def get(self, source, with_abstract=False):
        """get article using correspond source name"""
        getter = doc_getter(self.engine, self.Document, with_abstract)
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

def get_inner_annotation(annotation):
    """example: Optional[List[str]] -> Optional[str]; str -> str; Optional[str] -> Optional[str]"""
    if get_origin(annotation) is Union:
        for anno in get_args(annotation):
            if anno in [str, int, float, bool]:
                return annotation
            elif get_origin(anno) is list:
                return Optional[get_args(anno)[0]]
    else:
        return annotation if get_origin(annotation) is not list else get_args(annotation)[0]

def get_result_field_annotation(annotation, field_table):
    if get_origin(annotation) is Union:
        non_none_args = [arg for arg in get_args(annotation) if arg is not None] 
        return Optional[get_result_field_annotation(non_none_args[0], field_table)] # since sqlmodel does not suppport non-optional union as field type]
    else:
        return list[field_table] if get_origin(annotation) is list else field_table
    
class ResultDB(DB):
    """provide functionality for creating database, saving data, searching through database"""

    def __init__(self, engine, result_pydantic: BaseModel):
        self.engine = engine
        self.result_pydantic = result_pydantic
        self.NewBase = get_new_sql_base()
        self.created_models = self.create_sqlmodel_from_pydantic()
        self.Document, self.Result = self._define_sqltable()

    # def _define_sqltable(self) -> tuple[SQLModel, SQLModel]:
    #     NewDocBase = create_model(
    #         'NewDocBase', __base__=self.NewBase, **DOCBASE_DEF
    #     )
    #     NewResultBase = create_model(
    #         'NewResultBase', __base__=self.NewBase, **RESULT_DEF
    #     )
    #     ExtendResultBase = extend_fields(NewResultBase, self.result_pydantic)
        
    #     class Doc(NewDocBase, table=True):
    #         __tablename__ = 'documents'

    #         results: list['Result'] = Relationship(back_populates='document')
        
    #     class Result(ExtendResultBase, table=True):
    #         document: Doc = Relationship(back_populates='results')
        
    #     Doc = cast(SQLModel, Doc)
    #     Result = cast(SQLModel, Result)
    #     return Doc, Result
    
    def _define_sqltable(self):
        NewDocBase = create_model(
            'NewDocBase', __base__=self.NewBase, **DOCBASE_DEF
        )
        NewResultBase = create_model(
            'NewResultBase', __base__=self.NewBase, **RESULT_DEF
        )
        class Document(NewDocBase, table=True):
            __tablename__ = 'documents'
            results: list['Result'] = Relationship(back_populates='document')
        Result = self.complete_result_sqlmodel(NewResultBase)
        Document = cast(SQLModel, Document)
        Result = cast(SQLModel, Result)
        return Document, Result

    def create_db(self):
        """invocate `SQLModel.metadata.create_all`"""
        self.NewBase.metadata.create_all(self.engine)

    def get(self, source):
        """get article using correspond source name"""
        getter = doc_getter(self.engine, self.Document)
        return getter(source)
    
    # def save_result(self, text: str, metadata: dict[str, str], results: list[BaseModel]):
    #     """save text with extracted resutls"""
    #     assert super().check_source(metadata), 'metadata must have source field'
    #     assert isinstance(results[0], self.result_pydantic), f'model mismatch, expect {self.result_pydantic}'
    #     with Session(self.engine) as session, session.begin():
    #         document = self.Document(page_content=text, meta=metadata)
    #         for result in results:
    #             params = dict(result)
    #             result_sqlmodel = self.Result(**params)
    #             document.results.append(result_sqlmodel)
    #         session.add(document)
        
    def save_result(self, text: str, metadata: dict[str, str], results: list[BaseModel]):
        """called on the `Writer` object, save text with extracted resutls"""
        assert super().check_source(metadata), 'metadata must have a field named source'
        with Session(self.engine) as session, session.begin():
            document = self.Document(page_content=text, meta=metadata)
            for result in results:
                kw_args = self.construct_arguments_from_result(result)
                result = self.Result(**kw_args)
                document.results.append(result)
            session.add(document)


    def construct_arguments_from_result(self, result:BaseModel):
        kw_args = {}
        pydantic_fields = dict(result)
        for field_name, field in pydantic_fields.items():
            if field is None:
                continue
            if isinstance(field, list):
                kw_args[field_name] = [self.created_models[field_name](**{field_name: item}) for item in field]
            else:
                kw_args[field_name] = self.created_models[field_name](**{field_name: field})
        return kw_args


    def create_sqlmodel_from_pydantic(self):
        fields = self.result_pydantic.model_fields
        created_models = {}

        for field_name, field in fields.items():
            model_name = field_name.capitalize()
            table_name = f'result_{field_name}'
            # field_type = field.annotation if get_origin(field.annotation) is not list else get_args(field.annotation)[0]
            field_type = get_inner_annotation(field.annotation)
            default = field.default if not field.is_required() else ...

            created_model = create_model(
                model_name,
                __base__=self.NewBase,
                __cls_kwargs__={'table': True},
                __tablename__=table_name,
                id=(Optional[int], Field(default=None, primary_key=True)),
                result_id=(Optional[int], Field(default=None, foreign_key='results.id')),
                **{field_name: (field_type, Field(default))},
                result=(Optional['Result'], Relationship(back_populates=field_name))
            )
            # setattr(created_model, 'result', Relationship(back_populates=field_name))
            created_models[field_name] = created_model

        return created_models
    
    def complete_result_sqlmodel(self, base_model: Type[SQLModel]):
        # created_models = create_sqlmodel_from_pydantic(pydantic_model)
        fields = self.result_pydantic.model_fields
        field_defs = {}
        for field_name, field in fields.items():
            # field_type = list[created_models[field_name]] if get_origin(field.annotation) is list else created_models[field_name] 
            field_type = get_result_field_annotation(field.annotation, self.created_models[field_name]) # return list[field_table] or field_table
            field_defs[field_name] = (field_type, Relationship(back_populates='result'))
        field_defs['document'] = ('Document', Relationship(back_populates='results')) # seems that `Relationship` object cannot be defined in the resultbase, so we need to define it here
        
        result_model = create_model(
            'Result',
            __base__=base_model,
            __cls_kwargs__={'table': True},
            __tablename__='results',
            **field_defs
        )
        return result_model


NewBase = get_new_sql_base()
class ExtractRecord(NewBase, table=True):
    __tablename__ = 'extract_record'
    id: Optional[int] = Field(None, primary_key=True)
    key: str = Field(..., index=True)
    namespace: str
    # namespace is used to isolated different extracted process


class ExtractManager:
    """the primary goal was to skip those articles which have been extracted. Design inspiration is originated from langchain index method.
    sync version support presently.
    """

    def __init__(self, namespace, db_url):
        """
        __init__ 

        Args:
            namespace (str): the name to this extraction task, used for distinguish purpose.
            db_url (str): where you store your extract result
        """
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
