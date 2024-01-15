# Brochure
### What is _sisyphus_ for?
- Get articles and their SI in fast speed, all you need is provide the DOIs and have a nice connection with internet (be sure that you have accessed promission to the articles)
- Extract any information you want (delivered in text table <sup>in development</sup> form from the article), just defining some prompt to interact with llms (support openai models e.g., chatGPT 3.5)

### Pre-requisite
- An openai API key
- Elsevier API key (optional)
- Good internet connection

### The main API
- Download articles:
  -- file: *main_crawler.py*. Note that you can export DOIs from [web of science](https://webofscience.clarivate.cn/wos/alldb/basic-search/).
  
  `python main_crawler.py --retrieval_dois <file_contains_doi>`
- Extract data, output as json format
    -- file: *main.py*. Please pay attention to your prompt, which should be consistent with the corpus of the article. Enable pydantic support for more control with the output.

  `python main.py`

### Contribution
- Please first give your suggestions or any problems in the issue and then considering pull request.
- The main logic applied to the extraction process are followed by embedding, classification, summarization. Developers can consider how to optimize those procedures.

#### Tips
- Where you can adujust crucial prameters
  - in *sisyphus/manipulator/__init__.py* creating_jsonl() function -> change the chunk_size (this controls the size of each sentence to be extracted) to find the best outcome for your system.
  - in *sisyphus/crawler/download_si.py* download() function -> change the spawn rate (control the rate of the reqeust speed) to be compatible with your internet speed.

