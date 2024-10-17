import inspect
import logging

from .logger import LogLevel, LogFactory, set_global_log_level, get_logger

def log(name : str = None) -> logging.Logger:
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
    return get_logger(name=name)

def setLogLevel(level: LogLevel):
    set_global_log_level(level)

__all__ = ['setLogLevel', 'LogLevel', 'log', 'func']
