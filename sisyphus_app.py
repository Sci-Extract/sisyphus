"""
The app was inspired by project created by CharlyWargnier https://github.com/streamlit/example-app-zero-shot-text-classifier/tree/main
"""
import json

import streamlit as st
import pandas as pd

from sisyphus.utils.utilities import log


logger = log("test_log.txt")

# constants
EXTRACT_LOCATION = "data_articles"

# streamlit config
st.set_page_config(page_title="sisyphus", page_icon="⛏️")

c1, c2 = st.columns([0.32, 2])

with c1:
    st.image(
        "streamlit/images/sisyphus_myth.jpg",
        width=100,
    )
with c2:
    st.caption("")
    st.title("sisyphus extractor")

# sidebar
st.sidebar.write("")

# For elements to be displayed in the sidebar, we need to add the sidebar element in the widget.

# We create a text input field for users to enter their API key.

API_KEY = st.sidebar.text_input(
    "Enter your OpenAI API key",
    help="Get it from https://platform.openai.com/api-keys",
    type="password",
)

st.sidebar.markdown("---")

st.sidebar.write(
    """

App created by [Soike](https://github.com/sukiluvcode) using [Streamlit](https://streamlit.io/).

"""
)

# functions
def construct_prompt(dataframe):
    """
    schema of the prompt:
    the result json should be consistent with following structure, the outmost key is "compounds",
    the value of compounds are a list of compound, for each compound, the keys are {attributes_sep_by_space}
    , and the 
    """
    pass

# Tabs
MainTab, InfoTab = st.tabs(["Main", "Info"])

# init session state dict
session_state_cache = ("finish", "enable_pydantic", "enable_test")
for key in session_state_cache:
    if key not in st.session_state:
        st.session_state[key] = False

session_text_cache = ("query", "classify", "summarise", "output_json")
for key in session_text_cache:
    if key not in st.session_state:
        st.session_state[key] = None

with InfoTab:

    st.markdown("""
### What is _sisyphus_ for?
- Get articles and their SI in fast speed, all you need is provide the DOIs and have a nice connection with internet (be sure that you have accessed promission to the articles)
                
> [!NOTE]
> Presently, article retrieval is only accessible in CLI mode.
                
- Extract any information you want (delivered in text or table <sup>in development</sup> form from the article), just defining some prompt to interact with llms (support openai models e.g., chatGPT 3.5)

### Pre-requisite
- An openai API key
- Elsevier API key (optional)
- Good internet connection

### Main API
- Download articles:
  -- file: *main_crawler.py*. Note that you can export DOIs from [web of science](https://webofscience.clarivate.cn/wos/alldb/basic-search/).
  
  `python main_crawler.py --retrieval_dois <file_contains_doi>`
- Extract data, output as json format
    -- file: *main.py*. Please pay attention to your prompt, which should be consistent with the corpus of the article. Enable pydantic support for more control with the output.

  `python main.py`

### Contribution
- Please first give your suggestions or any problems in the issue and then considering pull request.
- The main logic applied to the extraction process are followed by embedding, classification, summarization. Developers can consider how to optimize those procedures.

####  Note
- *sisyphus_app* is the UI for sisyphus project, for more information, check on my [github page](https://github.com/sukiluvcode).
                """)

with MainTab:
    st.write("Follow instructinos below to extract out data you want 😎")
    st.write("")

    # options
    enable_pydantic = st.checkbox('Enable pydantic', key='pydantic', value= st.session_state["enable_pydantic"], help='enable for more fine control over result')
    st.session_state["enable_pydantic"] = enable_pydantic

    test_mode = st.checkbox('Test mode',key='test_mode', value=st.session_state["enable_test"], help='switch on to enable test mode.')
    st.session_state["enable_test"] = test_mode
    if test_mode:
        import os
        maximum = 0
        for publisher in os.listdir(EXTRACT_LOCATION):
            maximum += len(os.listdir(os.path.join(EXTRACT_LOCATION, publisher)))
        test_size = st.slider('test size', 1, maximum)

    with st.form("work_flow"):
        st.markdown("""#### 😶‍🌫️Locate""")
        st.markdown("Construct a semantic similiar sentnce with the target.")

        # query
        query = st.text_area(
            'Query',
            placeholder='Try to give a descriptional sentence',
            value=st.session_state["query"],
            help="the words should no less than 50 and no more than 200",
            key='1'
        )
        st.session_state["query"] = query
        st.write("")
        
        # classify
        st.markdown("#### 🤓Classify")
        st.markdown("Define criteria to evaluate the validaty of these sentences")
        classify = st.text_area(
            'Classify prompt',
            placeholder='Give some criterions to validate these sentences',
            value=st.session_state["classify"],
            help='The stricter the rules are, the more precise (~=less) the result you may obtain.',
            key='2'
        )
        st.session_state["classify"] = classify
        st.write("")
        
        submitted = st.form_submit_button("Submit/Save draft")
        
        error = False
        
        # validate user's api key
        if submitted:
            if not API_KEY.startswith('sk-'):
                st.markdown(":eyes: **it seems that your API key is invalid**")
                error = True
            # validate user's input
            if not query:
                st.markdown(":eyes: **it seems that you haven't insert a query.**")
                error = True
            if not classify:
                st.markdown(":eyes: **it seems that you haven't insert a classify prompt.**")
                error = True
                error = True
            if error:
                st.stop()
        
        
        if submitted:
            import asyncio
            import time
            from dotenv import find_dotenv, load_dotenv

            from sisyphus.processor.llm_extraction import Extraction
            from sisyphus.utils.utilities import get_format_instructions
            
            st.session_state['finish'] = False
            query, prompt_cls= query, classify
            if "pydantic_model" not in st.session_state:
                st.warning("no pydantic model found, please define it in data model.")
                st.stop()
            model = st.session_state.pydantic_model
            prompt_sum = get_format_instructions(model)
            logger.info(prompt_sum)
            st.stop()

            # execute code
            _ = load_dotenv(find_dotenv())
            system_message = "You are reading a piece of text from chemistry articles about nonlinear optical (nlo) materials and you are required to response based on the context provided by the user."
            d = dict(query=query, prompt_cls=prompt_cls, prompt_sum=prompt_sum, system_message=system_message)
            start = time.perf_counter()
            extraction = Extraction(from_=EXTRACT_LOCATION, save_filepath="Extraction_data.jsonl", query_and_prompts=d, embedding_limit=(5000, 1000000), completion_limit=(5000, 80000), max_attempts=5, logging_level=10)
            asyncio.run(extraction.extract(sample_size=test_size, pydantic_model=model))
            end = time.perf_counter()
            st.markdown(f"cost {end - start} s")
                    
            st.success("✅ Done!")
            st.session_state["finish"] = True

# if st.session_state["finish"]:
#     from script.result_extract import show_csv
#     df = show_csv("not decide")
#     st.dataframe(df)
#     csv = df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="Download data as CSV",
#         data=csv,
#         file_name="llm_extract.csv",
#         mime='text/csv'
#     )