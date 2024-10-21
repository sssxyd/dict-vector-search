import logging
import os
from enum import Enum
from logging.handlers import TimedRotatingFileHandler

from .func import get_executable_directory


class LogLevel(Enum):
    FATAL = 50
    ERROR = 40
    WARN = 30
    INFO = 20
    DEBUG = 10
    ALL = 1
    UNSET = 0


global_log_level: LogLevel = LogLevel.INFO


def set_global_log_level(level: LogLevel):
    level = LogLevel.INFO if level == LogLevel.UNSET else level
    global global_log_level
    global_log_level = level


def get_global_log_level():
    return global_log_level


def get_log_level(level: str) -> LogLevel:
    level = level.strip().upper()
    if level == "FATAL" or level == "CRITICAL":
        return LogLevel.FATAL
    if level == "ERROR":
        return LogLevel.ERROR
    if level == "WARN" or level == "WARNING":
        return LogLevel.WARN
    if level == "INFO":
        return LogLevel.INFO
    if level == "DEBUG":
        return LogLevel.DEBUG
    return global_log_level


def get_logger(name: str, level: LogLevel = LogLevel.UNSET, file_name: str = 'app', line_number : bool = True) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log

    log_dir = os.path.join(get_executable_directory(), 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file_path = os.path.join(log_dir, f"{file_name}.log")
    log_level = global_log_level if level == LogLevel.UNSET else level

    log_format_str = "%(asctime)s"
    if name and line_number:
        log_format_str += " - %(name)s"
    if line_number:
        if name:
            log_format_str += ":%(filename)s:%(lineno)d"
        else:
            log_format_str += " - %(filename)s:%(lineno)d"
    if level != LogLevel.ALL:
        log_format_str += " - [%(levelname)s]"
    log_format_str += " - %(message)s"

    formatter = logging.Formatter(log_format_str)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    info_handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1)
    info_handler.setLevel(level=LogLevel.DEBUG.value if log_level == LogLevel.ALL else log_level.value)
    info_handler.suffix = "%Y%m%d"
    info_handler.setFormatter(formatter)
    log.addHandler(info_handler)

    log.setLevel(level=LogLevel.DEBUG.value if log_level == LogLevel.ALL else log_level.value)

    return log


class LogFactory:
    @staticmethod
    def getLogLevelValue(level: str) -> LogLevel:
        return get_log_level(level=level)

    @staticmethod
    def setDefaultLogLevel(level: LogLevel):
        set_global_log_level(level=level)

    @staticmethod
    def getDefaultLogLevel() -> LogLevel:
        return get_global_log_level()

    @staticmethod
    def getLog(name: str, level: LogLevel = LogLevel.UNSET, file_name: str = 'device') -> logging.Logger:
        log_level = get_global_log_level() if level == LogLevel.UNSET else level
        return get_logger(name=name, level=log_level, file_name=file_name)
