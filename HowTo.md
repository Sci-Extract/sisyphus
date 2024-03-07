### Deployment
Copy this repo
```
>>> git clone https://github.com/sukiluvcode/sisyphus.git <your_project_name>
```

Install dependencies
```
>>> pip -r requirements.txt
```
After install, run
```
>>> playwright install
```

Add .env file to your root directory, which should include
- OPENAI_API_KEY
- els_api_key (Elsevier api key, used for article retrieval, Optional)

### Usage
The common work flow for sisyphus includes:
1. Prepare a file containing DOIs. Get it from WOS or any resource you prefer.
2. Get those articles by
```
>>> python main_crawler.py --retrieval_dois <your_doi_file>
```
3. Convert downloaded files to plain text for later usage, in command line, typing
```
>>> .\parse_html.bat
```
4. Open sisyphus UI, in command line, typing
```
>>> streamlit run sisyphus_app.py
```
5. Inside the UI, Create your model first in the data model page (If you already defined it in your root, skip this step. Remember to name it as gen_pydantic.py). Then input your query and classify prompt at homepage and set a test size to running the extract process.