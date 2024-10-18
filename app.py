import asyncio
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

import basic.func
from basic import LogLevel
from constants import APP_NAME, APP_VERSION
from service import aiModel
from service import dictWords
from service import vectorIndex

# 初始化全局变量
words : dict[str, dictWords.DictWord] = dictWords.load_dict_word_set()
codes : list[set[str]] = dictWords.load_index_codes()
model: Optional[SentenceTransformer] = aiModel.load_sentence_transformer_model()
word_index, pinyin_index = vectorIndex.load_vector_indexes()
info : dict[str, Any] = {
    "name": APP_NAME,
    "version": APP_VERSION,
    "loadTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "sentence-transformers/distiluse-base-multilingual-cased-v1",
    "dictWordSize": len(words),
    "indexWordSize": len(codes),
}
create_index_lock = asyncio.Lock()
# 全局线程池，最多可以创建一定数量的线程
executor = ThreadPoolExecutor(max_workers=10)

# 初始化日志记录器
access_logger = basic.log(file_name="access.log", level=LogLevel.ALL)
error_logger = basic.log(file_name="error.log", level=LogLevel.ALL)

# 使用 async contextmanager 创建 lifespan 事件处理器
@asynccontextmanager
async def lifespan(_: FastAPI):
    # 应用启动时
    access_logger.info(f"{info['name']} - V{info['version']} started")
    access_logger.info(f"Service info: {info}")

    yield  # 这里会继续运行应用的主循环

    # 应用关闭时
    access_logger.info(f"{info['name']} - V{info['version']} stopped")

app = FastAPI()

# 使用 @app.middleware("http") 来实现中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()

    # 获取真实的客户端 IP
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # 获取 X-Forwarded-For 中的第一个 IP，它是客户端的真实 IP
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        # 如果没有 X-Forwarded-For，直接使用 request.client.host
        client_ip = request.client.host

    access_logger.info(f"Request: {client_ip} - {request.method} - {request.url}")
    response = await call_next(request)
    end_time = datetime.now()
    duration = (end_time - start_time).microseconds / 1000
    access_logger.info(
        f"Response: {client_ip} - {response.status_code} - {request.method} - {request.url} - {duration:.3f}ms")
    return response


# 捕获并记录错误日志
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    # 获取真实的客户端 IP
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # 获取 X-Forwarded-For 中的第一个 IP，它是客户端的真实 IP
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        # 如果没有 X-Forwarded-For，直接使用 request.client.host
        client_ip = request.client.host

    error_logger.error(f"Error: {client_ip} - {exc} - Path: {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred."},
    )


async def _create_vector_index_and_reload(ai_model: SentenceTransformer, reload: bool, main_pid: int, worker: int,
                                         ngram_min: int, ngram_max: int, batch_size: int):
    try:
        keys = dictWords.prepare_index_words(ngram_min, ngram_max)
        vectorIndex.create_vector_indexes(keys, ai_model, worker, batch_size)
        if reload:
            os.kill(main_pid, signal.SIGHUP)
    except Exception as e:
        basic.log().error(msg="创建索引失败", exc_info=e)
    finally:
        create_index_lock.release()

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

    # 尝试获取锁，设置最大等待时间为 500 ms
    try:
        await asyncio.wait_for(create_index_lock.acquire(), timeout=0.5)
    except asyncio.TimeoutError:
        return {'code': 103, 'msg': "其他任务正在创建索引，请等待其结束后重试", 'micro': basic.cost_macro(micro_start)}

    # 异步创建索引
    main_pid = os.getppid()     #  Gunicorn 主进程ID
    executor.submit(_create_vector_index_and_reload, model, reload, main_pid, worker, ngram_min, ngram_max, batch_size)

    return {'code': 1, 'message': 'success', 'micro': basic.cost_macro(micro_start)}

@app.get("/reload")
async def reload_via_signal():
    micro_start = datetime.now()
    os.kill(os.getppid(), signal.SIGHUP)  # 给 Gunicorn 主进程发送 SIGHUP 信号
    return {'code': 1, 'message': 'success', 'micro': basic.cost_macro(micro_start)}

@app.get("/search")
async def search_vector_index(word : str, top : int = 3, pinyin : bool = False):
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
