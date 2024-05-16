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

# New update at 5/13
在 Indexing 之后，只需要在 test_chain.py 脚本文件下，定义 pydantic model 以及 filter function (nullable)，再添加上几个 examples 用以辅助大语言模型进行学习。最后修改 `create_all` 函数中的对应参数即可。
> pydantic 定义指导：对于挖掘过程中不确定是否存在的值，建议在 pydantic 定义中添加上默认值 None (or any default value make sense), '...'表示不给默认值。注意不要定义嵌套模型，也就是 pydantic field 指向另一个 pydantic 模型，
或者是定义field 为列表类型比如，list[float], 如果确实有需求，请联系作者。

# New update at 5/16
主要更新  
- 可创建不需词向量嵌入的数据库，入口文件为`tet_plain_db.py`， 创建后，对应的，在`test_chain.py`文件中修改 #166 `db`变量
- 文章元数据可以扩展，例如增加url等
- 对于存放挖掘结果数据库的查询支持

修改
- 文章元数据改为json格式，以便适应可拓展性