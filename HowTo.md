# Quick tutorial
- downloading process and parsing process omitted here

## Indexing
```
# before this remember to start chromadb docker container
python test_aindex.py -d <directory which contains parsed html> -c <name of the database>
```

## Extracting
```
python test_chain.py -d <directory which contains parsed html> -c <name of the database> -q "single-component adsorption isotherms, uptake value are 3 mmol g-1 at 298 K 1 bar" [-b <batch size, default 10>]
```
> NOTE: -b is optional, increase it to speed up process and query is modifiable

# 更改说明：
_test\_chain.py_ 文件中，可以自定义的是filter以及pydantic model，在定义完成你自己的pydantic model后，请及时修改 _update\_resultbase_ 以及 _create\_chain_ 中的实参