import psutil
import uvicorn
from gunicorn.app.base import BaseApplication

import basic.func
from constants import APP_VERSION, SERVER_PORT, APP_NAME

# 导入必要的依赖，防止pyinstaller打包时未能正确识别
import fastapi
import starlette
import pypinyin
import jieba
import torch
import sentence_transformers
import faiss
import numpy
import pydantic
import multipart

class UvicornGunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.app = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.app


def run_gunicorn_with_preload(server_port : int, process_worker : int):
    from app import app
    options = {
        "bind": f"0.0.0.0:{server_port}",  # 绑定地址和端口
        "workers": process_worker,  # 设置worker数量
        "preload_app": True,  # 预加载应用
        "worker_class": "uvicorn.workers.UvicornWorker",  # 使用 Uvicorn worker
    }
    UvicornGunicornApp(app, options).run()

    # windows下，只能使用uvicorn，不能使用gunicorn，只开一个进程，不作为生产环境使用
    # uvicorn.run(
    #     app = app,  # 这里是你的 FastAPI 实例的位置
    #     host = "0.0.0.0",         # 监听所有网络接口
    #     port = server_port,              # 端口号
    #     workers = 1,  # Windows下，只开一个进程
    # )


if __name__ == "__main__":
    args = basic.func.load_args()
    if 'version' in args or 'Version' in args or 'v' in args or 'V' in args:
        print(f"{APP_NAME} - {APP_VERSION}")
        exit(0)

    if 'help' in args or 'Help' in args or 'h' in args or 'H' in args:
        print(f"Usage: vector-search --version")
        print(f"Usage: vector-search [-port=8080] [-worker=0]")
        exit(0)

    port = int(args.get("port", SERVER_PORT))
    worker = int(args.get("worker", 0))
    worker = psutil.cpu_count(logical=False) if worker <= 0 else worker
    run_gunicorn_with_preload(server_port=port, process_worker=worker)
