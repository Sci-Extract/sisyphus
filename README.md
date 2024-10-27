# Step through _sisyphus_ extraction process
Author: Soike Yuan  date: 6/12/2024

Prerequisite: Assumed that you have installed sisyphus on your machine and add .env file to your root path which contains API key. :shipit:
> NOTE: all scripts are executed at root path.

## Overlook
Before we get start, I want to tell you the general workflow of sisyphus, which is,  
articles collection -> articles indexing -> define data schema -> extraction  
Usually, you need to first do some trial and error in the last  two steps, then apply the logic to a much larger corpus.  
btw, there is another thing, you can always execute python /a/script/file.py -h to know what parameters does it requires.
## Preparation stage
First, we need to download some articles by using crawler provided by sisyphus. Making sure that you are not using proxy(vpn)!!!   
`python main_crawler.py your/doi/file -r <rate>`  
You don't have to set the -r parameter, which default is set to 0.15, means that every 1/0.15 s make a request to one publisher.  
> IMPORTANT NOTE: Remember this is treated as attack behaviour for the publisher side, so decrease the total number of articles to be downloaded! If your crawler complains about the detection, set test parameter to 1, this may help!  

Ok, we collect those articles after running the crawler, good, that's a great begin! We can make it better by processing those raw html files, again in your command line, typing  
`python script/one_step_parse.py`  
If you are on windows system, use '\\' instead of '/'. Again, you can type python script/one_step_parse.py -h to know the parameters

## Indexing it
Good, we download and process those articles to a nice format, now it's time to open your proxy(vpn) to connect to the openai API. If you want to use the query functionality in later retrieval steps, following below.  
In your command line, run  
`python indexing.py -d /last/step/processed/dir -c <collection name> --local <0/1>`  
> NOTE: better fit the collection name to your extraction target, e.g., nlo. If you don't want to connect to the docker server which provide remote connection, then set --local to 1.  

When you don't need the query functionality in later process, run below.  
`python plain_indexing.py -d /last/step/processed/dir --db_name <your db name> --full_text <0/1>`  
> NOTE: better fit to your db name to extraction object, e.g., nlo. Set full_text to 1 to enable full text instead of chunking articles to small pieces (useful when your target is scatterd).

## Extraction
siuuuuuu! we made to the final step, but still the most important one.  
In this step, you need to write some code, but don't worry, nothing is that hard. Treat sisyphus as a normal one as other third-party packages, later, you will get what I means.  
Go through [chain](./chain.ipynb) notebook first.  

If you have read the jupyter notebook, then you probably want to know how to extract those articles at one time. Taking a look at the code snippet showing below
```python
import asyncio
from sisyhphus.chain import run_chains_with_extraction_history
...
define extraction chains
...
asyncio.run(run_chains_with_extraction_history(chain, <article_dir>, <batch_size>, <name_space>))
```
> Note: you may want to know why you have to provide article dir again since in last indexing step we already indexing them. In here, I use the article dir to get those article names in order to locate the articles in the vetor/plain database, and there is also an advantage, you can provide with a subset of the previous indexing files, which means that you can performing extraction only on those files. Batch size is number of articles to be processed in parallel, name space is the name to distinguish different extraction task, for example,  you can give name as 'nlo/shg'.

## Advanced usage
This is a bonus section, read if you want some high customizations of `Chain` object. 
In a word, sisyphus permit you to add user defined function to participate in any part of `Chain`, which means, you can not use the default chain = Filetr + Extractor + Validator + Writer, but to use what you created.  
Again, thanks to langchain, I borrowed this idea from its `RunnableLambda`.  
There are two scenario which I think might be useful to define your own chain than to overwrite implementation of sisyphus default chain. One is to modify the content to extract before the Extractor.
```python
# assumed you have defined your Filter, Extractor, Validator, Writer...
def modify_content(docs): # docs is a list of `Document` objects
    # change the original content
    for doc in docs:
        doc.page_content = doc.metadata.title + doc.page_content # for simplicity, I only add the title to the origin content
    return docs # You have to return this

# then your chain will looks like
chain_with_injection =  filter + modify_content + extractor + validator + writer # do not call modify_content, sisyphus will call it later!!!
```
And another scenario is to redirect extracted result instead of saving to database
```python
# assumed you have defined your Filter, Extractor, Validator...
# here, we just print out the results, you can do any things, really.
def print_results(docinfos): # docinfos is a list of `DocInfo` objects
    for docinfo in docinfos:
        print(docinfo.info) 
chain_without_writer = filter + extractor + validator + print_results
```

## Future development
Since the object of extraction is quite variaty, ranging from chem/physi properties to chemical reactions. It's hard to define a googd pipeline for a new task without futher tuning. For example, If I want to extract solid-state inorganic reactions from paper, for the prompt chaining method (which is now implemented by sisyphus), one could think that create a classification module and then the extraction module (module is an abstraction of LLM program). To achieve a promising output, every component needed to maintain in order to coordinate with each other, so, a lot of prompt engineering and expert knowledge required. It was brittle in other words.
DSPy is a tool to systematically to create a LLM program, it's do not require any prompt (well, mostly). Breifly, it's a tool to easily create promgram and make it to the best suite out of box.