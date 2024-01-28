import asyncio
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.prompts import ChatPromptTemplate


model = ChatOpenAI(temperature=0, model_kwargs={"response_format":{"type":"json_object"}}, model="gpt-3.5-turbo-1106")
json_parser = JsonOutputParser()
str_parser = StrOutputParser()

summarise_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, summarise the text provided by user"),
    (
        "human", "below are titile and abstract of an article from {topic} field in chemistry domain. "\
        "Based on the provided information, please identify the category of the material and giving the physical or chemical properties associated with it, such as band gap or permeability"\
        "Giving the results in below json format\n{{\"type\": , \"properties\": []}}\nhere is titile:\n{title}\nhere is abstract:\n{abstract}"
    )
])
truncate_prompt = ChatPromptTemplate.from_template(
    "I'll give you a list of properties"\
    "\nFor each item in the list, remove the part which acted as adjective, leave only the noun part. List:\n{properties}"\
    "\nGiving response in below json format\n{{\"properties\": []}}"
)
uniqueness_prompt = ChatPromptTemplate.from_template(
    "I'll give you a list of properties, among which, some of them actually means the same property, e.g., shg response, shg generation."\
    "\nPlease remove any redundant elements and provide a list of unique elements.. List:\n{properties}"\
    "\nGiving response in below json format\n{{\"properties\": []}}"
)
setup = {"properties": itemgetter("properties")}

truncate_chain = setup | truncate_prompt | model | json_parser
uniqueness_chain = setup | uniqueness_prompt | model | json_parser

concentration_chain = truncate_chain | uniqueness_chain
summarise_chain = summarise_prompt | model | json_parser

def merge_properties(summarise_output):
    properties = []
    for summarise in summarise_output:
        properties.extend(summarise["properties"])
    return {"properties": properties}

# read abstract file
import json
abstracts = []
with open("abstracts.jsonl", encoding='utf-8') as file:
    for line in file:
        abstracts.append(json.loads(line))

inputs = [
    {"title": abstract["title"],
     "abstract": abstract["abstract"],
     "topic": "nonlinear optical materials"}
     for abstract in abstracts
]

summarise_output = asyncio.run(summarise_chain.abatch(inputs))
print(merge_properties(summarise_output))
print(concentration_chain.invoke(merge_properties(summarise_output)))
