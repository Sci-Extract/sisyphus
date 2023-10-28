import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter


tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=15,
    length_function=tiktoken_len
)

def chunk_text(text: str) -> list[str]:
    strip_text = text.replace("\n", " ")
    texts = text_splitter.create_documents([strip_text])
    ret = [text.page_content for text in texts]
    return ret
