from bs4 import BeautifulSoup
from pathlib import Path
import tiktoken
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticToolsParser
from sisyphus.patch import achat_httpx_client


file = '10.1021&sl;acsami.1c25005.html'
soup = BeautifulSoup(Path(file).read_text(encoding='utf8'), features='lxml')
plain_text = soup.get_text(separator=' ', strip=True)

encoding = tiktoken.get_encoding('cl100k_base')


def ensure_safe_len(text, max_token=5000):
    encoding_text = encoding.encode(text)
    chunks = []
    full_length = len(encoding_text)
    for i in range(0, full_length, max_token):
        chunks.append(encoding.decode(encoding_text[i : i + 5000]))
    return chunks


class FindFullName(BaseModel):
    shorthand: str = Field(
        ..., description='the shorthand of a chemical compound name'
    )
    full_name: str = Field(
        ..., description='the full name or definition of the shorthand'
    )


def create_formatted_output_chain(output_model: BaseModel):
    model = ChatOpenAI(temperature=0, http_async_client=achat_httpx_client)
    model = model.bind_tools(tools=[output_model], tool_choice='auto')
    parser = PydanticToolsParser(tools=[output_model])
    chain = model | parser
    return chain


resolve_prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'Please use your language understanding capabilities to decode or explain the shorthand provided by the user.',
        ),
        (
            'user',
            'full text: {text}\ncontext: {para}.\n shorthand: {shorthand}, For each shorthand listed, please find its expansion or meaning within the input text.' 
            'If the expansion is not directly available, provide a relevant explanation or context based on your understanding.',
        ),
    ]
)
chain = resolve_prompt | create_formatted_output_chain(FindFullName)
import asyncio

para = 'The eternal porosities of selected 1 and 2 were established through N2 gas adsorption tests at 77 K. As obviously observed, 2 paradigm could adsorb a large amount of N2, yielding a N2 uptake of 280 cm3 g-1 at 298 K and 1 bar, which was slighter lower than that of 1 (310 cm3 g-1) (Figure 3a). The reduced N2 uptakes were mainly derived from the occupation of F and CH3 entities in the confined pore space which slightly reduced the pore volume. The derived pore distribution using Horvath-Kawazoe model also suggests an ultramicropore aperture of ca. 5.8 Å (SI Figure S7). We further collected the single-gas isotherms of gas molecule on 2 at 273 and 298 K up to 1 bar. As is clearly revealed in Figure 3b,c, 2 exhibited a distinguishing adsorption steepness and reversed adsorption of C2H6 over C2H4 at both temperatures. The preferential adsorption capacity of C2H6 over 2 was up to 3.06 mmol g-1 (68.5 cm3 g-1), being notably higher than that of C2H4 (2.28 mmol g-1) (51.1 cm3 g-1) at 298 K and 1 bar. This indicated that C2H6 with higher polarizability could be preferentially captured in the inert pore channel, confirming the unwonted reversed adsorption behavior. Such results were also further confirmed through testing at 273 K (Figure 3c). Combined with the assessed uptake ratio, 2 demonstrated a gratifying C2H6/C2H4 uptake ratio, with a larger value of 134% (SI Table S4) at 298 K and 1 bar, far exceeding the advanced benchmark materials including MUF-15 (113%), (1) ZJU-HOF-1 (121), (34) and LIFM-63 (130), (35) although being slightly inferior in comparison to the advanced ethane-selective Ni(bdc)(ted) (180%) (36) and Cu(Qc)2 (237%) (37) adsorbents.'
inputs = [
    {'shorthand': ('1', '2'), 'para': para, 'text': text}
    for text in ensure_safe_len(plain_text)
]


async def resolve_shorthand(shorthand: tuple, para: str, full_text: str):
    chunks = ensure_safe_len(full_text)
    resolve_chain = resolve_prompt | create_formatted_output_chain(
        FindFullName
    )
    for chunk in chunks:
        res = await resolve_chain.ainvoke(
            {'shorthand': shorthand, 'para': para, 'text': chunk}
        )
        if not res:
            print('not find')
            continue
        else:
            print(res)
            return res


res = asyncio.run(
    resolve_shorthand(shorthand=('1', '2'), para=para, full_text=plain_text)
)
