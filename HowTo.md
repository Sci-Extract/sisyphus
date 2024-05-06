# Quick totorial
- downloading process and parsing process omitted here

## Indexing
```
# before this remember to start chromadb docker container
python test_aindex.py -d <directory which contains parsed html> -c <name of the database>
```

## Extracting
`python test_chain.py -d <directory which contains parsed html> -c <name of the database> [-b <batch size, default 10>]`
> NOTE: -b is optional, increase it to speed up process.
