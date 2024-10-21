import inspect
import logging
from datetime import datetime

from .logger import LogLevel, LogFactory, set_global_log_level, get_logger

def log(name : str = None, level: LogLevel = LogLevel.UNSET, file_name: str = 'app', line_number : bool = True) -> logging.Logger:
    # if name is None:
    #     # 获取当前函数的上一个栈帧
    #     caller_frame = inspect.currentframe().f_back
    #     # 获取调用者的模块信息
    #     module = inspect.getmodule(caller_frame)
    #     if module:
    #         if hasattr(module, '__file__'):
    #             name = module.__file__
    #         elif hasattr(module, '__name__'):
    #             name = module.__name__
    #         else:
    #             name = ""
    #     else:
    #         name = ""
    return get_logger(name=name, level=level, file_name=file_name, line_number=line_number)

def cost_macro(start_time: datetime) -> int:
    return (datetime.now() - start_time).microseconds

def setLogLevel(level: LogLevel):
    set_global_log_level(level)

__all__ = ['setLogLevel', 'LogLevel', 'log', 'func']
