# 逐步完成 _sisyphus_ 提取过程
作者：Soike Yuan 日期：2024 年 6 月 12 日

先决条件：假设您已在机器上安装了 sisyphus，并将包含 API 密钥的 .env 文件添加到根路径。:shipit:
> 注意：所有脚本都在根路径下执行。

## 概览
在开始之前，我想告诉您 sisyphus 的一般工作流程，即  
文章收集 -> 文章索引 -> 定义数据模式 -> 提取  
通常，您需要先在最后两个步骤中进行一些反复试验，然后将逻辑应用于更大的语料库。  
顺便说一句，还有一件事，您可以随时执行 python /a/script/file.py -h 来了解它需要什么参数。
## 准备阶段
首先，我们需要使用 sisyphus 提供的爬虫下载一些文章。确保您没有使用代理（vpn）！！！  
`python main_crawler.py your/doi/file -r <rate>`  
您可以不设置 -r 参数，其默认设置为 0.15，这意味着每 1/0.15 秒向一个发布者发出一个请求  
好的，我们收集了这些文章，很好，这是一个很好的开始！我们可以通过处理这些原始 html 文件来使其变得更好，再次在命令行中输入  
`python script/one_step_parse.py`  
如果您使用的是 Windows 系统，请使用 '\' 而不是 '/'。同样，您可以输入 python script/one_step_parse.py -h 来了解参数

## 索引
很好，我们下载并处理了这些文章，并将其转换为良好的格式，然后是时候打开代理（vpn）以连接到 openai API 了。如果您想在以后的检索步骤中使用查询功能，请按照以下步骤操作。
在命令行中，运行
`python indexing.py -d /last/step/processed/dir -c <collection name> --local <0/1>`
> 注意：集合名称更适合您的提取目标，例如 nlo。如果您不想连接到提供远程连接的 docker 服务器，请将 --local 设置为 1。

当您在以后的过程中不需要查询功能时，请在下面运行。  
`python plain_indexing.py -d /last/step/processed/dir -db_name <your db name> --full_text <0/1>`
> 注意：db 名称更适合您的提取对象，例如 nlo。将 full_text 设置为 1 以启用全文，而不是将文章分块成小块（当您的目标分散时很有用）。

## 提取
siuuuuuu！最后一步，但也是最重要的一步。  
在此步骤中，你需要编写一些代码，但别担心，没有什么是那么难的。将 sisyphus 视为与其他第三方包一样的普通包，稍后您将明白我的意思。  
首先浏览 ./chain.ipynb 笔记本。

如果您已经阅读过 jupyter 笔记本，那么您可能想知道如何一次性批量提取文章。如下面的代码片段所示
```
import asyncio
from sisyhphus.chain import run_chains_with_extraction_history
...
define extract chains
...
asyncio.run(run_chains_with_extraction_history(chain, <article_dir>, <batch_size>, <name_space>))
```
> 注意：您可能想知道为什么您必须再次提供文章目录，即使在上一个索引步骤中我们已经对它们进行了索引。在这里，我使用 article dir 来获取这些文章名称，以便在 vector/plain 数据库中定位文章，这样做有一个好处，你可以提供以前的索引文件的子集，这意味着你可以只对这些文件进行提取。批量大小是需要并行处理的文章数量，名称空间是区分不同提取任务的名称，例如你可以将其命名为“nlo/shg”。  