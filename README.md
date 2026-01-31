# Step through _sisyphus_ extraction process
Author: Soike Yuan  date: 1/12/2025

Prerequisite: Assumed that you have installed sisyphus on your machine and add .env file to your root path which contains API key. :shipit:
> NOTE: all scripts are executed at root path.

## Install
python version == 3.10; use poetry as package management tool, run `poetry install`

## Below is the file structure of the project,

```
├─ README.md                        
├─ pyproject.toml                   # Python project metadata and dependencies
├─ data/                            # Extracted data of high entropy alloys
├─ notebooks/                       # Jupyter notebooks
├─ db/                              # sql databases
├─ record/                          # record sql database
│─ sisyphus/
│     ├─ chain/                     # chain implementation
│     └─ crawler/                   # crawler for article downloading
│     └─ heas/                      # high entropy alloys relevant prompt and code
│     └─ utils/                     # helpful functions
│─ pipeline                         # example here
│─ test_file                        # example article files
└─ script/
```

### More specifically
notebooks is where you should go to find examples code and can give it a try :) make sure you are executing in your root directory

### machine learning code
Refer to [Machine learning](https://github.com/sukiluvcode/Machine-learning-of-sisyphus-project)