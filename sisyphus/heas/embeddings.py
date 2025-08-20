from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
chroma_db = Chroma(collection_name='synthesis_embedding', embedding_function=embeddings)
has_embedded = []

QUERY_SYN = """Experimental procedures describing the synthesis and processing of HEAs materials, including methods such as melting, casting, rolling, annealing, heat treatment, or other fabrication techniques. Details often include specific temperatures (e.g., °C), durations (e.g., hours, minutes), atmospheric conditions (e.g., argon, vacuum), mechanical deformation (e.g., rolling reduction)."""
QUERY_STRENGTH = "The stress-strain curve of alloy, describes yield strength (ys), tensile strength (uts) and elongation properties, for example, CoCuFeMnNi shows tensile strength of 1300 MPa and total elongation of 20%."
QUERY_PHASE = """Microstructure characterization of alloys (common phases include FCC, BCC, HCP, L12, B2 etc.), usually through technique like XRD or TEM. Describe about phase and grain size and boundaries"""
K = 3

def retrieve(vector_store, source, query, sub_titles, k=K):
    if not sub_titles:
        filter_ = {"source": source}
    else:
        filter_ = {"$and":[{"sub_titles": {"$in": sub_titles}}, {"source": source}]}
    return vector_store.similarity_search(query, k=k, filter=filter_)

def match_subtitles(docs, pattern):
    sub_titles = list(set([doc.metadata["sub_titles"] for doc in docs]))
    target_titles = []
    for title in sub_titles:
        if pattern.search(title):
            target_titles.append(title)
    return target_titles

