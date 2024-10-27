# -*- coding:utf-8 -*-
"""
@File    :   validators.py
@Time    :   2024/05/24 15:30:11
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   store validators for common use case.
"""

import json
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

from sisyphus.chain.chain_elements import DocInfo
from sisyphus.utils.tenacity_retry_utils import openai_429_retry_wraps, pydantic_validate_retry_wraps


def base_validator(*params):
    """validators should comform with this schema, passing parameters which decided at run time"""
    def validator(to_validate: DocInfo):
        pass
    return validator


# You can customize your own prompt template used for validation process, but ensure include doc and info placeholder
RE_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ('system', 'Please verify the extracted results. Some of these results may not meet the requirements. Discard unqualified results based on user provided criterias and remain the qualified results unchanged.'),
    ('human', 'origin text: {doc}\nextracted results: {info}.'),
    MessagesPlaceholder('check_prompt'),
    ('human', 'Respond in JSON format, maintaining the structure of the extracted results. Use the format: {{"verified_results": [extracted_results]}}. '
     'Discard any results that do not meet the criteria. Only when there are no results meet the criteria, return: {{"verified_results": []}}'),
]
)

def get_parser(pydantic_model):
    def parser(message: AIMessage):
        """parser for parsing AI response json as pydantic objects"""
        json_response = json.loads(message.content)
        pydantic_objects = []
        if not json_response['verified_results']:
            return
        for r in json_response['verified_results']:
            pydantic_objects.append(pydantic_model(**r))
        return pydantic_objects
    return parser

def llm_validator(chat_model, pydantic_model, prompt_template: ChatPromptTemplate, user_defined_prompt):
    """call first, then get validator"""
    input_vars = prompt_template.input_variables
    required_vars = ['doc', 'info']
    assert all(var in input_vars for var in required_vars)
    user_var = list(set(input_vars).difference(set(required_vars)))
    parser = get_parser(pydantic_model)
    chain = prompt_template | chat_model.bind(response_format={'type': 'json_object'}) | parser

    @openai_429_retry_wraps
    async def call(input_):
        return await chain.ainvoke(input_)

    @pydantic_validate_retry_wraps
    async def validator(to_validate: DocInfo) -> DocInfo | None:
        """validate extracted results"""
        # TODO: handle pydantic parsing error
        extract_results = json.dumps([dict(result) for result in to_validate.info])
        validate_result = await call(
            {
                'doc': to_validate.doc.page_content,
                'info': extract_results,
                user_var[0]: user_defined_prompt
            }
        )
        if not validate_result:
            return
        result_with_doc = DocInfo(doc=to_validate.doc, info=validate_result)
        return result_with_doc
    
    return validator


# Extend this for user using case
CHEM_ABBRE_MAPPING = {
    'methane': 'CH4',
    'C1': 'CH4',
    'ethane': 'C2H6',
    'C2': 'C2H6',
    'ethylene': 'C2H4',
    'propane': 'C3H6', 
    'C3': 'C3H6'
}

def coercion_validator(coercion_fields: list[str]):
    """validator acted as normalizing the word form"""
    def validator(to_validate: DocInfo):
        extracted_results = [dict(result) for result in to_validate.info]
        for coercion_field in coercion_fields:
            for result in extracted_results:
                if (key:=result[coercion_field].lower()) in CHEM_ABBRE_MAPPING:
                    result[coercion_field] = CHEM_ABBRE_MAPPING[key]
        pydantic_model = type(to_validate.info[0]) # get pydantic model first
        validate_results = [pydantic_model(**result) for result in extracted_results]
        result_with_doc = DocInfo(doc=to_validate.doc, info=validate_results)
        return result_with_doc
    return validator
