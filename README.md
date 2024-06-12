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
You can not set the -r parameter, which default is set to 0.15, means that every 1/0.15 s make a request to one publisher.  
Ok, we collect those articles, good, that's a great begin! We can make it better by processing those raw html files, again in your command line, typing  
`python script/one_step_parse.py`  
If you are on windows system, use '\' instead of '/'. Again, you can type python script/one_step_parse.py -h to know the parameters

## Indexing it
Good, we download and process those articles to a nice format, then it's time to open your proxy(vpn) to connect to the openai API. If you want to use the query functionality in later retrieval steps, following below.  
In your command line, run  
`python indexing.py -d /last/step/processed/dir -c <collection name> --local <0/1>`  
> NOTE: collection name better fit to your extraction target, e.g., nlo. If you don't want to connect to the docker server which provide remote connection, then set --local to 1.  

When you don't need the query functionality in later process, run below.  
`python plain_indexing.py -d /last/step/processed/dir -db_name <your db name> --full_text <0/1>`  
> NOTE: db name better fit to your extraction object, e.g., nlo. Set full_text to 1 to enable full text instead of chunking articles to small pieces (useful when your target is scatterd).

## Extraction
siuuuuuu! the final step, but the most important one.  
In this step, you need to write some code, but don't worry, nothing is that hard. Treat sisyphus as a normal one as other third-party packages, later, you will get what I mean.  
Go through ./chain.ipynb notebook first.  

If you have read the jupyter notebook, then you probably want to know how to extract those articles at one time. As the code snippet showing below
```
import asyncio
from sisyhphus.chain import run_chains_with_extraction_history
...
define extraction chains
...
asyncio.run(run_chains_with_extraction_history(chain, <article_dir>, <batch_size>, <name_space>))
```
> Note: you may want to know why you have to provide article dir again since in last indexing step we already indexing them. In here, I use the article dir to get those article names in order to locate the articles in the vetor/plain database, and there is an advantage, you can provide with a subset of the previous indexing files, which means that you can performing extraction only on those files. Batch size is number of articles to be processed in parallel, name space is the name to distinguish different extraction task, for example,  you can give name as 'nlo/shg'.

