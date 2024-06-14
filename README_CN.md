# 逐步完成 _sisyphus_ 提取过程
作者：Soike Yuan 日期：2024 年 6 月 12 日

先决条件：假设您已在机器上安装了 sisyphus，并将包含 API 密钥的 .env 文件添加到根路径。:shipit:
> 注意：所有脚本都在根路径下执行。

## 概览
在开始之前，我想告诉您 sisyphus 的一般工作流程，即  
文章收集 -> 文章索引 -> 定义数据模式 -> 提取  
通常，您需要先在最后两个步骤中进行一些反复试验，然后将逻辑应用于更大的语料库。  
顺便说一句，您可以随时执行 python /a/script/file.py -h 来了解它需要什么参数。
## 准备阶段
首先，我们需要使用 sisyphus 提供的爬虫下载一些文章。确保您没有使用代理（vpn）！！！  
`python main_crawler.py your/doi/file -r <rate>`  
> 重要提示：请记住，对于发布者而言，这被视为攻击行为，因此请减少要下载的文章总数!（不超过200篇在一段时间内）如果您的爬虫程序抱怨被检测到，请将测试--test设置为 1，这可能会有所帮助！  

您可以不设置 -r 参数，其默认设置为 0.15，这意味着每 1/0.15 秒向一个发布者发出一个请求  
好的，使用爬虫后我们收集了这些文章，很好，这是一个很好的开始！我们可以通过处理这些原始 html 文件来使其变得更好，再次在命令行中输入  
`python script/one_step_parse.py`  
如果您使用的是 Windows 系统，请使用 '\\' 而不是 '/'。同样，您可以输入 python script/one_step_parse.py -h 来了解参数

## 索引
很好，我们下载并处理了这些文章，并将其转换为良好的格式，然后是时候打开代理（vpn）以连接到 openai API 了。如果您想在以后的检索步骤中使用查询功能，请按照以下步骤操作。
在命令行中，运行  
`python indexing.py -d /last/step/processed/dir -c <collection name> --local <0/1>`
> 注意：请选择符合挖掘目标的名字作为集合名称，例如 nlo。如果您不想连接到提供远程连接的 docker 服务器，请将 --local 设置为 1。

当您在以后的过程中不需要查询功能时，请在下面运行。  
`python plain_indexing.py -d /last/step/processed/dir -db_name <your db name> --full_text <0/1>`
> 注意：请选择符合挖掘目标的名字作为数据库名，例如 nlo。将 full_text 设置为 1 以启用全文，而不是将文章分块成小块（当您的目标分散时很有用）。

## 提取
siuuuuuu！最后一步，但也是最重要的一步。  
在此步骤中，你需要编写一些代码，但别担心，没有什么是那么难的。将 sisyphus 视为与其他第三方包一样的普通包，稍后您将明白我的意思。  
首先浏览 [chain](./chain.ipynb) 笔记本。

如果您已经阅读过 jupyter 笔记本，那么您可能想知道如何一次性批量提取文章。批量操作如下面的代码片段所示
```
import asyncio
from sisyhphus.chain import run_chains_with_extraction_history
...
define extract chains
...
asyncio.run(run_chains_with_extraction_history(chain, <article_dir>, <batch_size>, <name_space>))
```
> 注意：您可能想知道为什么您必须再次提供文章目录，即使在上一个索引步骤中我们已经对它们进行了索引。在这里，我使用 article dir 来获取这些文章名称，以便在 vector/plain 数据库中定位文章，这样做有一个好处，你可以提供以前的索引文件的子集，这意味着你可以只对这些文件进行提取。批量大小是需要并行处理的文章数量，名称空间是区分不同提取任务的名称，例如你可以将其命名为“nlo/shg”。  

## 高级用法
这是一个奖励部分，如果您想要对 `Chain` 对象进行一些更高级的自定义，请阅读。  
总之，sisyphus 允许您添加用户定义的函数来参与 `Chain` 的任何部分，这意味着，您可以不使用默认链 = Filetr + Extractor + Validator + Writer，而是使用您创建的链。  
再次感谢 langchain，我从它的 `RunnableLambda` 中借用了这个想法。  
我认为有两种情况可能更适合定义您自己的链，而不是覆盖 sisyphus 默认链的实现。一种是在 Extractor 之前修改要提取的内容。
```
# 假设您已经定义了 Filter、Extractor、Validator、Writer...
def modify_content(docs): # docs 是 `Document` 对象的列表
    # 更改原始内容
    for doc in docs:
        doc.page_content = doc.metadata.title + doc.page_content # 为简单起见，我仅将标题添加到原始内容
    return docs # 您必须返回此内容

# 然后您的链将看起来像
chain_with_injection = filter + modify_content + extractor + validator + writer # 不要调用modify_content，sisyphus 稍后会调用它！！！
```
另一种情况是重定向提取的结果而不是保存到数据库
```
# 假设您已经定义了 Filter、Extractor、Validator...
# 在这里，我们只是打印出结果，但您可以做任何事情，真的。
def print_results(docinfos): # docinfos 是 `DocInfo` 对象的列表
    for docinfo in docinfos:
        print(docinfo.info)
chain_without_writer = filter + extractor + validator + print_results
```