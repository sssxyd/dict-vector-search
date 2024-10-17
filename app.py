import asyncio
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
from starlette.responses import FileResponse

import basic.func
from service import aiModel
from service import dictWords
from service import vectorIndex

app = FastAPI()
words : dict[str, dictWords.DictWord] = dictWords.load_dict_word_set()
codes : list[set[str]] = dictWords.load_index_codes()
model: Optional[SentenceTransformer] = aiModel.load_sentence_transformer_model()
word_index, pinyin_index = vectorIndex.load_vector_indexes()
info : dict[str, Any] = {
    "name": "Vector-Search",
    "version": "1.0.1",
    "loadTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
    "dictWordSize": len(words),
    "indexWordSize": len(codes),
}
create_index_lock = asyncio.Lock()

async def create_vector_index_and_reload(ai_model: SentenceTransformer, reload: bool, main_pid: int, worker: int,
                                         ngram_min: int, ngram_max: int, batch_size: int):
    try:
        keys = dictWords.prepare_index_words(ngram_min, ngram_max)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, vectorIndex.create_vector_indexes, keys, ai_model, worker, batch_size)
        if reload:
            os.kill(main_pid, signal.SIGHUP)
    except Exception as e:
        basic.log().error(msg="创建索引失败", exc_info=e)

@app.get("/favicon.ico")
async def favicon_ico():
    return FileResponse('static/favicon.ico')

@app.get("/")
async def index_html():
    return FileResponse('static/index.html')

@app.get("/info")
async def get_service_info():
    micro_start = datetime.now()
    result = info.copy()
    result["dictWordLastModifyTime"] = dictWords.get_dict_words_last_modify_time()
    result["indexWordLastModifyTime"] = dictWords.get_index_words_last_modify_time()
    result["wordIndexLastModifyTime"] = vectorIndex.get_word_index_last_modify_time()
    result["pinyinIndexLastModifyTime"] = vectorIndex.get_pinyin_index_last_modify_time()
    return {'code': 1, 'message': 'success', 'result': result, 'micro': basic.cost_macro(micro_start)}

@app.post("/put")
async def upload_dict_words(file: UploadFile = File(...)):
    micro_start = datetime.now()
    # 检查文件类型
    if not file.filename.endswith(".csv"):
        return {'code' : 100, 'msg' : f"上传的文件[{file.filename}]必须是 CSV 格式", 'micro': basic.cost_macro(micro_start)}

    # 如果需要更加严格的检查，还可以检查 content_type
    if file.content_type != "text/csv":
        return {'code': 101, 'msg': f"文件MIME类型[{file.content_type}]不匹配，必须是 text/csv", 'micro': basic.cost_macro(micro_start)}

    file_location = os.path.join(basic.func.get_executable_directory(), 'dict', 'dict_words.csv')
    basic.func.touch_dir(file_location)
    if os.path.exists(file_location):
        os.remove(file_location)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
    return {'code': 1, 'message': 'success', 'result': basic.func.get_file_last_modify_time(file_location),
            'micro': basic.cost_macro(micro_start)}

@app.get("/index")
async def create_vector_index(ngram_min : int = 3, ngram_max : int = 5, worker : int = 0, batch_size = 500, reload: bool = False):
    micro_start = datetime.now()
    # 尝试获取锁
    if create_index_lock.locked():
        return {'code': 102, 'msg': "索引创建已在进行，请稍后重试", 'micro': basic.cost_macro(micro_start)}
    main_pid = os.getppid()     #  Gunicorn 主进程ID
    async with create_index_lock:
        asyncio.create_task(create_vector_index_and_reload(ai_model=model, reload=reload, main_pid=main_pid,
                                                           worker=worker, ngram_min=ngram_min, ngram_max=ngram_max,
                                                           batch_size=batch_size))
    return {'code': 1, 'message': 'success', 'micro': basic.cost_macro(micro_start)}

@app.get("/reload")
async def reload_via_signal():
    micro_start = datetime.now()
    os.kill(os.getppid(), signal.SIGHUP)  # 给 Gunicorn 主进程发送 SIGHUP 信号
    return {'code': 1, 'message': 'success', 'micro': basic.cost_macro(micro_start)}

@app.get("/search")
async def search_vector_index(word : str, top : int = 5, pinyin : bool = False):
    micro_start = datetime.now()
    if not word:
        return {'code': 103, 'msg': "搜索词不能为空", 'micro': basic.cost_macro(micro_start)}
    if not word_index or not pinyin_index:
        return {'code': 104, 'msg': "索引尚未创建，请先reload", 'micro': basic.cost_macro(micro_start)}
    if not model:
        return {'code': 105, 'msg': "模型尚未加载，请先reload", 'micro': basic.cost_macro(micro_start)}

    results = vectorIndex.search_vector_indexes(word=word, model=model, word_index=word_index,
                                                pinyin_index=pinyin_index, index_codes=codes, dict_words=words,
                                                top_k=top, pinyin=pinyin)
    return {'code': 1, 'message': 'success', 'result': results, 'micro': basic.cost_macro(micro_start)}
