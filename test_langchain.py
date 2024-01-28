import asyncio
import json
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def reduce_summary(from_file: str):
    map_prompt = ChatPromptTemplate.from_template("You are reading an abstract cite from a chemistry article in material domain. Summarise the abstract into one concise sentence, "\
                                            "while the content should include the material and its relevant properties. here is the title: {title} and abstract:\n{abstract}. Helpful answer:")
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
    output_parser = StrOutputParser()

    inputs = []
    with open(from_file, encoding='utf-8') as file:
        for line in file:
            info = json.loads(line)
            inputs.append(info)

    chain = {"abstract": itemgetter("abstract"), "title": itemgetter("title")} | map_prompt | model | output_parser
    map_response = asyncio.run(chain.abatch(inputs))

    def concatenate(responses: list[str]):
        return "\n".join(responses)

    collective_response = concatenate(map_response)
    reduce_chain = (
        ChatPromptTemplate.from_template(
            "The following is set of summaries: \n{collective_response}"
            "\nBased on the summaries, please give an consolidated response. The answer should begin with \"The articles are mainly talking about...\""
        )
        | model
        | output_parser
    )

    finnal_summary = reduce_chain.invoke({"collective_response": collective_response})
    return finnal_summary
