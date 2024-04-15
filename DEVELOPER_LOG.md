# This developer log is devoted to give coding directions in the future
_Since the version 1.0 was completed in recent days, I decide to write this to give some instructions for myself, in case that I will forget due to some uncertain causes_

## Compatibility
This is the issue which can't be ignored. LangChain has become one of the most popular LLMs relevant packages, meanwhile, its coding style is so simple to chain all the crucial components, not limited to embedding, chunking, prompt templates, output parse... All these features can achieved by LCEL, which was inspired by the linux pipeline, so niche! Without further talk, here comes to the main point: I need to change the present logic to be compatible with LangChain, while the request should be throttled to fully exploited the quota of my account.
> I love speed

## Model agnostic
Well well well, this project should not be restricted to openai model only, just extending to more models, maybe google claude or something like it...

## During my investigation of nlo field articles
After I manually check serval papers from Pan, I realise that the paper can be categorised (if I want to introduce the full pipeline, I need to consider about category, but can be skipped). There are papers talking about a new devised nlo crystal, then in the abstract section giving its most outstanding properties, like band gap, bire, shg, cut off, LIDT. More rarely, some paers just talking about a new type of material, but not mention about the previous mentions properties. While some papers are talking about the strategy or using the theretical way to explore the candidates nlos, for which I should filter, however there are some papers talking about strategy but gived the strategy by synthesis nlo, then talking about some of the properties.

## some thoughts
* I have considered to just use the whole article for data extraction, there are two reason which stop me to do so, one is the price is around 2.5 times of the original method and the other is that this may bring hallucination and obscure results.
* During the search, I found there are still some paper's doi are not included in my crawler engine. Maybe I can expand the old logic to widen the doi acceptance. (what if the doi affiliation is not subscribe???)
* The SI format usually be one of .pdf(more) or .docx, sometimes, abundant info are stored in tables, this become a reason why i need to donwload SI. As for the parse method, I should really considered about it.
* I should mention in the prompt that the calculated value is not needed
- Using the text embeddings and the categories to train a model, using randomforest. I just thinking that it might become one of the results section for comparison, used as text classification. [link]("https://github.com/openai/openai-cookbook/blob/main/examples/Classification_using_embeddings.ipynb") 

## about the paper
The main content focuses on the full pipeline of text mining. The overall task is divided into subtasks, then devising a chain to connect those subtasks together. I think the role LLM plays is that previous tasks can be replaced or enhanced by LLM, such as classification and information extraction. Meanwhile, I should note that LLM is a generative model which might produce hallucinated answers, so it's crucial for the validation process (the validation itself is actually part of the chain).