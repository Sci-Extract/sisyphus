# From chaos to clarity: Designing a material data extraction chain enhanced by LLM for versatile Scenario adaptation

## Structure of the paper
The main content focuses on the full pipeline of text mining. The overall task is divided into subtasks, then devising a chain to connect those subtasks together. I think the role LLM plays is that previous tasks can be replaced or enhanced by LLM, such as classification and information extraction. Meanwhile, I should note that LLM is a generative model which might produce hallucinated answers, so it's crucial for the validation process (the validation itself is actually part of the chain).

## Schedule of coming week
Finish the text classification section. 3 methods.

## Introduction

Data driven material design has become one of the most popular field in material field, leading to an increased focus on database construction, while material literature serving as a promising resource for data extraction.  
Literature are consisted of unstructured data, e.g., natural language, images and semi-structured data like table. Those data are hard to interpreted by machines. There are many tools dedicated to extracting data from scientific domain, e.g., ChemicalTagger, ChemDataExtrctor, BioBERT(maybe need brief illustration for each tool), however, there still have a considerable labor cost in the preparing stage, i.e., constructing dictionary consisted of target properties and lack of generalizability. Recently, the emergence of large language models (LLMs) has possessed the capability of understanding human language and responding based on the input text (prompt). Unsurprisingly, one of the biggest use case is in extraction.  
Prompt engineering is a relatively new field for developing and optimizing prompts to enhance their suitability across diverse application contexts. Considering that the response given by LLMs is non-deterministic and tends to hallucinate information, it is necessary to build a standard workflow to ensure the reliability of the response. In the context of data extraction in material fields, there are two main categories of extraction, i.e., synthesis parameters and physical or chemical properties of material. With the prompt engineering technique, language chains, we can divide the extraction task to sub-tasks and solve them sequentially.
Herein, we developed a toolkit named _sisyphus-extractor_ using ChatGPT-3.5 model combine with language chain to successfully extracted data from material domain.
> If I need to point out advantage against before papers (JACS, NC), I will refer to the flexibility of change langchain corresponds to different working scenario.

## Implementation
> TL;DR  
The implementation details are trivial. For this draft I will draw a flowchart instead.

### Overview
This toolkit provide an end-to-end data extraction pipeline, which takes HTML format articles as input and generates a database hosted by MongoDB. The methods used in each stage are described below.
### Article retrieving
We use the publisher provided API and web scraping tools for article retrieving.
- A corpus of 387,878 articles from 33 different journals was generated using a series of purpose-built web-scraping tools.
### Article processing
The article processing steps include removing HTML syntax and dividing into chunks (around 200 words for each).
### Two steps filetering
First, for each article, the chunks are embedded as vectors using openAI text-embedding-ada-002 model, then calculating the context similarity for chunks with a predefined query. Second, highly relevant chunks from previous step are passed to a classify prompt for futher validation, only these results labeled as True passed to next step.  
... following steps are omited

## Results and discussion
> There are two main tasks, one is experimental parameter extraction, another is property extraction. We should use the tool to extract serval instances (maybe NLO properties and MOF synthesis extraction) to showcase the ability.
Outline of this section:  
1. Calculate precision, recall, F1 score for model evaluation
2. Create a database agent using LLM to chat with your data.
3. Use the generated property database combine with some structure descriptors as ML training set, to predict a property, e.g., nlo shg value.


## important parameters
- query used for search: nonlinear optical crystals (year from 2015 to date) (article only) (materials science)
    - from: [web of science](https://webofscience.clarivate.cn/)


- [ ] github markdown