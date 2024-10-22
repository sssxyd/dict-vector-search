# dict-vector-search
本项目利用SentenceTransformer对字典词条进行向量化，创建Faiss向量索引，实现错别字、同音字、形似字、漏字等情况的搜索词召回。

## Example
- 单氨胖光酸 <code>http://localhost:8080/search?word=单氨胖光酸&top=1&pinyin=1</code>
    ```json
    {
      "code": 1,
      "message": "success",
      "result": [
        {
          "index": "WORD",
          "code": "15273",
          "word": "甘草酸单铵半胱氨酸氯化钠注射液",
          "score": 5,
          "distance": 0.375848472118378
        }
      ],
      "micro": 110706
    }
    ```
- 阿代那非 <code>http://localhost:8080/search?word=阿代那非&top=1&pinyin=1</code>
    ```json
    {
      "code": 1,
      "message": "success",
      "result": [
        {
          "index": "WORD",
          "code": "24317",
          "word": "阿伐那非片",
          "score": 7,
          "distance": 0.395916491746902
        }
      ],
      "micro": 154205
    }
    ```
- 霜瓜唐安<code>http://localhost:8080/search?word=霜瓜唐安&top=1&pinyin=1</code>
    ```json
    {
      "code": 1,
      "message": "success",
      "result": [
        {
          "index": "PINYIN",
          "code": "4598",
          "word": "双瓜糖安胶囊",
          "score": 4,
          "distance": 0.0730657055974007
        }
      ],
      "micro": 129010
    }
    ```
## Usage

### 创建/更新字典
- `dict/dict_words.csv`：字典文件，每行一个词条
- 包含两个字符串字段，分别为: 词条代码、词条内容
- 可通过http(post): /put 接口上传字典文件
    ```shell
    curl -X POST -F "file=@xxx.csv" http://localhost:8080/put
    ```
  
### 创建索引
- 查看帮助
    - 源码运行
        ```shell
        python main.py -h
        ```
    - 可执行文件运行
        ```shell
        cd /path/to/vector-search
        ./vector-search -h
        ```
- 创建索引
  - 源码运行
      ```shell
      python main.py index -worker=4 -batch=1000 -min=2 -max=4
      ```
  - 可执行文件运行
      ```shell
      cd /path/to/vector-search
      ./vector-search index -worker=4 -batch=1000 -min=2 -max=4
      ``` 
    
### 启动服务
- 启动服务
    - 源码运行
        ```shell
        python main.py server -port=8080 -log-level=info
        ```
    - 可执行文件运行
        ```shell
        cd /path/to/vector-search
        ./vector-search server -port=8080 -log-level=info
        ```

- 访问帮助页面
    - <code>http://localhost:8080/</code>

## 安装打包
- 创建虚拟环境
    ```shell
    python -m venv .venv
    source venv/bin/activate
    ```
- 安装依赖
    ```shell
    pip install -r requirements.txt
    ```
- 下载预训练模型
    ```python
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    model.save('model/distiluse-base-multilingual-cased-v1')
    ```
- 打包程序
    ```shell
    pyinstaller ./vector-search.spec
    ```
  
## 注意事项
- 本例使用CPU，有GUP的情况下，可使用GPU加速，请修改faiss-cpu为faiss-gpu
