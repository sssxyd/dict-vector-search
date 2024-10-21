import os
import re
import sys
import time
from datetime import datetime
from typing import Any


def get_executable_directory():
    if getattr(sys, 'frozen', False):  # 判断是否为打包后的可执行文件
        executable_path = os.path.realpath(sys.executable)  # 获取实际可执行文件的路径
        directory = os.path.dirname(executable_path)  # 获取实际可执行文件所在的目录
    else:
        directory = os.path.dirname(os.path.realpath(__file__))  # 获取脚本文件所在的目录
        directory = os.path.dirname(directory)
    return directory


def load_args() -> dict[str, Any]:
    params = dict()
    for i in range(1, len(sys.argv)):
        arg = str(sys.argv[i])
        if arg.startswith('--'):
            params[arg[2:]] = True
        elif arg.startswith('-'):
            idx = arg.find('=')
            if idx > 0:
                params[arg[1:idx].strip()] = arg[idx + 1:].strip()
            else:
                params[arg[1:]] = None
        else:
            idx = arg.find('=')
            if idx > 0:
                params[arg[:idx].strip()] = arg[idx + 1:].strip()
            else:
                params[arg] = None
    return params

def get_duration(start_time: datetime) -> str:
    # 获取当前时间
    now = datetime.now()

    # 计算时间差
    duration = now - start_time

    # 提取天、秒、小时、分钟
    days = duration.days
    seconds = duration.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    micros = duration.microseconds
    mills = micros // 1000

    # 构建人类友好的字符串
    parts = []
    if days > 0:
        parts.append(f"{days}天")
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0:
        parts.append(f"{minutes}分钟")
    if seconds > 0:
        if mills >= 500:
            seconds = seconds + 1
        parts.append(f"{seconds}秒")
    if seconds == 0 and mills > 0:
        if micros - mills * 1000 >= 500:
            mills = mills + 1
        parts.append(f"{mills}毫秒")
    if mills == 0 and micros -  mills * 1000 > 0:
        parts.append(f"{micros -  mills * 1000}微秒")

    return ''.join(parts)


def is_http_url(url):
    """
    判断给定的字符串是否是有效的 HTTP 或 HTTPS URL。

    :param url: 待检查的字符串
    :return: 如果字符串是有效的 HTTP 或 HTTPS URL，则返回 True，否则返回 False
    """
    # 正则表达式匹配 HTTP 或 HTTPS URL
    pattern = re.compile(
        r'^https?://'  # http:// 或 https://
        r'([a-zA-Z0-9-]+\.)+'  # 子域名
        r'[a-zA-Z]{2,}'  # 顶级域名
        r'(/\S*)?'  # 可选路径
        r'$'
    )
    return bool(pattern.match(url))


def resolve_path(path):
    """
    判断给定的路径是绝对路径还是相对路径，并将相对路径转换为绝对路径。

    :param path: 待检查的路径字符串
    :return: 绝对路径字符串
    """
    # 判断是否为绝对路径
    if os.path.isabs(path):
        # 如果已经是绝对路径，则直接返回
        return path
    else:
        # 如果是相对路径，则使用 os.getcwd() 获取当前工作目录，并将其与相对路径拼接
        current_dir = os.getcwd()
        absolute_path = os.path.join(current_dir, path)
        return absolute_path


def touch_dir(path):
    # 分离文件扩展名，判断是否为文件路径
    dir_path, file_extension = os.path.splitext(path)

    # 如果有扩展名，表示是文件路径，获取父级目录
    if file_extension:
        # 只创建文件所在的父目录
        dir_path = os.path.dirname(path)
    else:
        # 没有扩展名，则按路径创建所有目录
        dir_path = path

    # 创建目录
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        print(f"创建路径 {dir_path} 时发生错误: {e}")

def get_file_last_modify_time(filepath : str) -> str:
    if not os.path.exists(filepath):
        return ""
    # 获取最后修改时间
    modification_time = os.path.getmtime(filepath)
    # 将时间格式化为可读形式
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modification_time))