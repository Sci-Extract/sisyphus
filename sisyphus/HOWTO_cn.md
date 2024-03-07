### 配置
克隆到本地文件
```
>>> git clone https://github.com/sukiluvcode/sisyphus.git <your_project_name>
```

安装依赖
```
>>> pip -r requirements.txt
```
运行完毕后，执行
```
>>> playwright install
```

根目录下添加 .env 文件，需要包括
- OPENAI_API_KEY
- els_api_key (爱思唯尔出版社的API， 可选)

### 使用
sisyphus通常的操作步骤包括如下
1. 准备好一个包含DOIs的文件，你可以从WOS或者其他地方获取
2. 下载文献，命令行中输入
```
>>> python main_crawler.py --retrieval_dois <your_doi_file>
```
3. 将下载的html格式转化为纯文本格式
```
>>> .\parse_html.bat
```
4. 打开sisyphus的用户界面
```
>>> streamlit run sisyphus_app.py
```
5. 在用户界面里，先在 data model 下创建你的数据模型（如果你已经使用pydantic定义了你的模型，跳过此步，并请将其命名为 gen_pydantic.py）。接着在主界面输入查询字符串以及用于判别的prompt，设置好测试挖掘文献的数目，开始运行，输出结果保存在UI_extract.jsonl下。