# Release
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

# New update at 6/2
modification: _fix some bugs, throttler implementation, elsevier parsing error_  
新增（按时间）
- 添加web-ui逻辑，`document`
- 调整进度条显示逻辑，现在更直观了，可以直接看到总共任务
- integrate cpp parser 
- add extract manager, used when unstable connection
- add validator (now has two default, coercion validator and re-check validator, check it in sisyphus/chain/validators)
- refactor code structure, make it more oop.

关于新版本`Chain`对象使用介绍(详细代码参考 test_oop_implementation.py):   
```
# 使用utils模块帮手函数获得文章数据库（向量或者不含向量），结果数据库，聊天模型的对象 
...
chat_model = get_chat_model('gpt-3.5-turbo'/'gpt-4o')
db = get_remote_chromadb(<collection_name>)
result_db = get_create_resultdb(<db_name>, <pydantic_model>)
...
# 实例化Filter, Extractor, Validator, Writer类
filter = Filter(<db>, <query>/None, <filter_function>)
extractor = Extractor(chat_model, <pydantic_object>, <examples>)
validator = Validator()
validor.add_gadget(<your_function_or_default_function>) # 可以添加多个validator函数,详情见 sisyphus/chain/validators
writer = Writer(result_db)

# 使用 '+' 创建`Chain`对象, 顺序很重要！
chain = filter + extractor + validator + writer

# 运行chain
asyncio.run(chain.acompose(<file_name>))
# or 运行多条chains
asyncio.run(run_chains_with_extraction_history(chain, <article_dir>, <batch_size>, <name_space>))
```

## patch at 6/5
add full text loader, support full text indexing.

## patch at 6/12
add sync support for chain, add local storage for chroma embedding, update README.md as documentation.  
- set logging level to 30 to enble task bar displaying.

## patch at 6/13
add a script to easy get the left dois when crawler program exit unexpected, check script/get_left_dois.py for more information