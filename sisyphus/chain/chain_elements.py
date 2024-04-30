# -*- coding:utf-8 -*-
'''
@File    :   chain_elements.py
@Time    :   2024/04/29 20:06:00
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   Defnition of Parser, Filter, Extractor, Validator, sqlWriter
'''

import asyncio
import os
import logging
import logging.config
from typing import Optional, Callable

from langchain_community.vectorstores import chroma
from langchain_core.pydantic_v1 import BaseModel, ValidationError
from langchain_core.documents import Document
from langchain.output_parsers import PydanticToolsParser
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from sisyphus.patch import ChatOpenAIThrottle


logging.config.fileConfig(os.sep.join(['config', 'logging.conf']))
logger = logging.getLogger('debugLogger')

class BaseElement(object):
    def __repr__(self):
        return self.__class__.__name__

class Filter(BaseElement):
    """Applied to Article level"""
    def __init__(self, db: chroma.Chroma, file_name: str, query: Optional[str] = None, filter_func: Optional[Callable] = None): # TODO: db parameter support plain database using sqlite  
       self.db = db
       self.file_name = file_name
       self.query = query
       self.filter_func = filter_func

    def locate(self) -> list[Document]:
        """select list of `Document` object in an article based on creterions, e.g., semantic similarity or use all"""
        self.check_database()
        if self.query:
            docs = self.db.similarity_search(query=self.query, filter={'source': self.file_name})
        else:
            docs = self.db._collection.get(where=self.file_name)
            # convert raw docs to `Document`
            raise NotImplementedError('not implement direct filter on doc right now!')
        return docs
    
    async def alocate(self) -> list[Document]:
        """async version, overwrite this if has native async method"""
        return await asyncio.to_thread(self.locate)

    def filter(self, doc: Document) -> bool:
        """filter on document, return boolean"""
        return self.filter_func(doc) if self.filter_func else True
    
    async def afilter(self, doc:Document) -> bool:
        return await asyncio.to_thread(self.filter, doc)
    
    def check_database(self) -> bool:
        """raise if query is given but without using chroma as database"""
        if self.query:
            if not isinstance(self.db, chroma.Chroma):
                raise ValueError(
                    f'{type(self.db)} type database does not support semantic filter, set query to None'
                    'or use Chroma database')


DEFAULT_PROMPT_TEMPLATE= ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant to extract information from text, if you are do not know the vlaue of an asked field, return null for the value. Call the function multiple times if needed'),
        MessagesPlaceholder(variable_name='examples'),
        ('human', '{text}')
    ]
)

class Extractor(BaseElement):
    """Applied to `Document` object, use openai tool call"""
    def __init__(self, chat_model: ChatOpenAIThrottle, pydantic_models: list[BaseModel], examples: list[BaseMessage]):
        self.chat_model = chat_model 
        self.pydantic_models = pydantic_models
        self.examples = examples
        self.prompt = DEFAULT_PROMPT_TEMPLATE
        self.pydantic_models = pydantic_models
        self.parser = PydanticToolsParser(tools=[self.pydantic_models])
    
    def build_chain(self):
        chain = self.prompt | self.chat_model.bind_tools(tools=self.pydantic_models, tool_choice='auto') | self.parser
        input_vars = self.prompt.input_variables
        if set(input_vars) != set(["examples", "text"]):
            raise ValueError(f"prompt template arguments inconsistent with receive, receive: [examples, text], expect: {input_vars}")
        return chain
 
    def extract(self, doc: Document) -> Optional[list[BaseModel]]:
        """extract info from doc"""
        chain = self.build_chain()
        result = None
        try: # add logic to handle openai rate limit error, process validation error
            result = chain.invoke({"examples": self.examples, "text": doc.page_content()})
        except ValidationError as e:
            logger.debug(e)
        return result

    async def aextract(self, doc: Document) -> Optional[list[BaseModel]]:
        """async version for extracting info from doc"""
        chain = self.build_chain()
        result = None
        try: # add logic to handle openai rate limit error, process validation error
            result = await chain.ainvoke({"examples": self.examples, "text": doc.page_content()})
        except ValidationError as e:
            logger.debug(e)
        return result

class Validator(BaseElement):
    """Applied to article level"""
    def validate(self, to_validate: list[BaseModel]):
        # TODO resolve demonstrative pronouns
        return to_validate
    
    async def avalidate(self, to_validate: list[BaseModel]):
        return await asyncio.to_thread(self.validate, to_validate)
    
class sqlWriter(BaseElement):
    pass
