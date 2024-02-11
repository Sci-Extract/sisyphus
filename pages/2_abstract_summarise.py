import asyncio
import json

import pandas as pd
import streamlit as st

from sisyphus.crawler.integral_search_fetch import construct_query, fetch
from test_langchain import reduce_summary

# basic config
st.set_page_config(
    page_title="utilities",
    page_icon="🧐",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Well, this utilities are designed for giving more inspirations for people"
    }
)

# instantiate session variable
if "search_query" not in st.session_state:
    st.session_state["search_query"] = None
if "search_complete" not in st.session_state:
    st.session_state["search_complete"] = False
if "summary" not in st.session_state:
    st.session_state["summary"] = ""

# constants
fetch_savepath = "abstracts.jsonl"

# sidebar
st.sidebar.write("")
st.sidebar.markdown(
    """🙌 App is built on top of [langchain](https://github.com/langchain-ai/langchain)"""
)

st.title("Fetch & Insight")
st.markdown("*Exploit the power of LLM, extract topics from articles.*")

fetch_tab, insight_tab = st.tabs(["Fetch", "Insight"])

# parser function, convert user input to params
def construct_parser(*, subjarea, pubyear, doctype, raw_keywords):
    """get the parameters for construct_query function"""
    match subjarea:
        case "chemistry":
            subjarea = "chem"
        case "physical":
            subjarea = "phys"
        case "material":
            subjarea = "mate"
        case "biology":
            subjarea = "bioc"
    match doctype:
        case "article":
            doctype = "ar"
        case "review":
            doctype = "re"
    keywords = raw_keywords.split()
    return pubyear, subjarea, doctype, keywords


with fetch_tab:
    st.markdown(
        """Please provide the topics you want to get insight from. *Special thanks to the scopus search API*.  
        *tips*: if you are familiar with scopus search language, feel free to construct it on your own 🤞🤞  
        *note*: retrieving 10 articles by default."""
    )
    options = st.radio(
        "Preference",
        ["construct by constructor", "construct by myself"],
        index=None
    )
    if options:
        if options == "construct by constructor":
            sub_cl, pub_cl, doc_cl = st.columns(3)
            with sub_cl:
                subjarea = st.radio(
                    "subject",
                    options=["chemistry", "physical", "material", "biology"]
                )
            with pub_cl:
                pubyear = st.slider(
                    "published from date (by year)",
                    min_value=1990,
                    max_value=2024
                )
            with doc_cl:
                doctype = st.radio(
                    "document type",
                    options=["article","review"]
                )
            raw_keywords = st.text_area(
                "Key words",
                placeholder="heart attack acute",
                help="please use space as separator"
            )
            construct_query_button = st.button(
                label="construct"
            )
            search_query = None
            if construct_query_button and raw_keywords:
                params = construct_parser(subjarea=subjarea, pubyear=pubyear, doctype=doctype, raw_keywords=raw_keywords)
                search_query = construct_query(*params)
                st.session_state["search_query"] = search_query
                st.markdown(f"Search query: {search_query}")
            
            if query := st.session_state["search_query"]:
                fetch_button = st.button(
                label="fetch"
                )
                if fetch_button:
                    asyncio.run(fetch(query))
                    st.markdown(f"save file in {fetch_savepath}.")
                    st.session_state["search_complete"] = True
            
        elif options == "construct by myself":
            search_query = st.text_area(
                "DIY"
            )
            fetch_button = st.button(label="fetch")
            if search_query and fetch_button:
                asyncio.run(fetch(search_query))
                st.markdown("save file in {fetch_savepath}.")
                st.session_state["search_complete"] = True

    if st.session_state["search_complete"]:
        title_abstract = []
        with open(fetch_savepath, encoding='utf-8') as file:
            for line in file:
                title_abstract.append(json.loads(line))
        st.json(title_abstract)
        # df = pd.DataFrame(data=title_abstract)
        # st.dataframe(df)

with insight_tab:
    st.markdown(
        f"""😁Methodology used here is called Map-Redude.  
        🥳To execute this process please ensure you have a stable connection with **openai API**.  
        *Extract from: {fetch_savepath}*"""
    )
    get_insight_button = st.button(label="Get insight")
    if get_insight_button:
        with st.spinner('Wait for it...'):
            summary = reduce_summary(from_file=fetch_savepath)
            st.session_state["summary"] = summary
        st.success('Done!')
    st.markdown('---')
    st.markdown(st.session_state["summary"])
        
