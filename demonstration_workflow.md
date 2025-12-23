## Before starting workflow
download articles from web using the script ...

### workflow

- parse articles  
```python script/one_step_parse.py```

- indexing  
```python plain_indexing.py -d <name of the dir where articles are saved> --db_name <name of the stored databas> --full_text <0/1>```

- labeling and extraction: refer to `pipeline.ipynb` file


