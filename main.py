import os.path
from datetime import datetime
from zipfile import error

import psutil
import uvicorn

import basic.func
from basic import LogFactory
from service import dictWords
from service import aiModel
from service import vectorIndex
from constants import APP_VERSION, SERVER_PORT, APP_NAME

# 导入必要的依赖，防止pyinstaller打包时未能正确识别
import sys
import fastapi
import starlette
import pypinyin
import jieba
import torch
import scipy
import sentence_transformers
import faiss
import numpy
import pydantic
import multipart


def run_uvicorn(server_port : int, log_level : str = "info"):
    from app import app
    server_log_level = LogFactory.getLogLevelValue(log_level)
    LogFactory.setDefaultLogLevel(server_log_level)
    uvicorn.run(
        app = app,  # 这里是你的 FastAPI 实例的位置
        host = "0.0.0.0",         # 监听所有网络接口
        port = server_port,              # 端口号
        workers = 1,  # Windows下，只开一个进程
        log_level = server_log_level.value,  # 日志级别
    )

def run_index(process_worker : int = 0, ngram_min : int = 3, ngram_max : int = 5, batch_size : int = 500):
    start_time = datetime.now()
    model = aiModel.load_sentence_transformer_model()
    if model is None:
        print("Failed to load sentence transformer model")
        return
    batch_index_dir = os.path.join(basic.func.get_executable_directory(), 'index', datetime.now().strftime("%Y%m%d%H%M%S"))
    print(f"Prepare index words to {batch_index_dir}")
    words = dictWords.prepare_index_words(batch_index_dir, ngram_min=ngram_min, ngram_max=ngram_max)
    print(f"Index words count: {len(words)}")
    vectorIndex.create_vector_indexes(batch_index_dir=batch_index_dir, index_words=words, model=model, worker=process_worker, batch_size=batch_size)
    print(f"Indexing completed in {basic.func.get_duration(start_time)}")

def run_usage():
    print(f"Usage: vector-search version")
    print("")
    print(f"Usage: vector-search server [-port=8080] [-log-level=info]")
    print(f"\t port: server port, default 8080")
    print(f"\t log-level: log level, default info")
    print("")
    print(f"Usage: vector-search index [-worker=0] [-min=3] [-max=5] [-batch=500]")
    print(f"\t worker: process worker count, default 0 means cpu count")
    print(f"\t min: ngram min length, default 3")
    print(f"\t max: ngram max length, default 5")
    print(f"\t batch: batch size for embeddings , default 500")

if __name__ == "__main__":
    args = basic.func.load_args()
    if 'version' in args or 'Version' in args or 'v' in args or 'V' in args:
        print(f"{APP_NAME} - {APP_VERSION}")
        sys.exit(0)

    if 'help' in args or 'Help' in args or 'h' in args or 'H' in args:
        run_usage()
        sys.exit(0)

    if 'index' in args or 'Index' in args:
        worker = int(args.get("worker", 0))
        min_gram = int(args.get("min", 3))
        max_gram = int(args.get("max", 5))
        batch = int(args.get("batch", 500))
        run_index(process_worker=worker, ngram_min=min_gram, ngram_max=max_gram, batch_size=batch)
        sys.exit(0)

    if 'server' in args or 'Server' in args:
        port = int(args.get("port", SERVER_PORT))
        level = args.get("log-level", "info")
        run_uvicorn(server_port=port, log_level=level)
    else:
        run_usage()