import csv
import os
import re
import time

import jieba
from pypinyin import pinyin, Style

import basic

class DictWord:
    def __init__(self, code: str, word: str):
        self.code = code
        self.word = word

    def __str__(self):
        return f"code={self.code}, word={self.word}"

    def __repr__(self):
        return self.__str__()


def trim_word(word):
    # 使用正则表达式匹配所有非字母数字的字符
    cleaned_word = re.sub(r'\W', '', word)
    return cleaned_word

def pinyin_word(word: str) -> str:
    result = pinyin(word, style=Style.NORMAL)
    pinyin_str = ''
    for item in result:
        pinyin_str += item[0] + ' '
    return pinyin_str.strip()

def split_word(word: str, ngram_min : int = 3, ngram_max : int = 5) -> set[str]:
    sub_words = set()

    # 对每个可能的子字符串长度进行循环
    for length in range(ngram_min, ngram_max + 1):
        for i in range(len(word) - length + 1):
            sub_words.add(word[i:i + length])

    # 使用结巴分词对中文进行分词
    for w in jieba.cut_for_search(word):
        if len(w) > 1:
            sub_words.add(w)
    return sub_words

def prepare_dict_words():
    log = basic.log()
    directory = os.path.join(basic.func.get_executable_directory(), 'dict')
    basic.func.touch_dir(directory)
    words_set = set()
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 只处理 .txt 文件
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            # 打开文件，读取每一行并将其添加到 set 中
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    # 移除行尾的换行符并去除多余空白
                    word = trim_word(line.strip())
                    if word:  # 确保行不为空
                        words_set.add(word)
            log.info(f">>loaded {len(words_set)} words from {filename}")
    log.info(f">>total {len(words_set)} words loaded")
    word_list = list(words_set)
    word_list.sort()
    filepath = os.path.join(basic.func.get_executable_directory(), 'dict', 'dict_words.csv')
    if os.path.exists(filepath):
        os.remove(filepath)
    row_index = 0
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 逐行写入数据
        for word in word_list:
            dw = DictWord(row_index, word)
            writer.writerow([dw.code, dw.word])
            row_index = row_index + 1
    log.info(f">>saved {len(word_list)} words to {filepath}")


def load_dict_words() -> list[DictWord]:
    log = basic.log()
    filepath = os.path.join(basic.func.get_executable_directory(), 'dict', 'dict_words.csv')
    if not os.path.exists(filepath):
        log.error(f"File not found: {filepath}")
        return []
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        words = []
        # 逐行读取数据
        for row in reader:
            dw = DictWord(row[0], row[1])
            words.append(dw)
    return words

def load_dict_word_set() -> dict[str, DictWord]:
    log = basic.log()
    filepath = os.path.join(basic.func.get_executable_directory(), 'dict', 'dict_words.csv')
    if not os.path.exists(filepath):
        log.error(f"File not found: {filepath}")
        return dict()
    words = dict()
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 逐行读取数据
        for row in reader:
            dw = DictWord(row[0], row[1])
            words[dw.code] = dw
    return words

def prepare_index_words(ngram_min : int = 3, ngram_max : int = 5) -> list[str]:
    log = basic.log()
    words = load_dict_words()
    keys = []
    index_words = dict()
    for word in words:
        sub_words = split_word(word.word, ngram_min, ngram_max)
        for sub_word in sub_words:
            if sub_word not in index_words:
                index_words[sub_word] = set()
            index_words[sub_word].add(word.code)
    filepath = os.path.join(basic.func.get_executable_directory(), 'index', 'index_words.csv')
    basic.func.touch_dir(os.path.dirname(filepath))
    if os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 逐行写入数据
        for key, value in index_words.items():
            keys.append(key)
            codes = ', '.join(str(num) for num in value)
            writer.writerow([key, codes])
    log.info(f">>saved {len(index_words)} index words to {filepath}")
    return keys

def load_index_codes() -> list[set[str]]:
    log = basic.log()
    filepath = os.path.join(basic.func.get_executable_directory(), 'index', 'index_words.csv')
    if not os.path.exists(filepath):
        log.error(f"File not found: {filepath}")
        return []
    codeList = []
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 逐行读取数据
        for row in reader:
            codes = set()
            for code in row[1].split(','):
                codes.add(code.strip())
            codeList.append(codes)
    return codeList

def load_index_word_codes() -> (list[str], list[set[str]]):
    log = basic.log()
    filepath = os.path.join(basic.func.get_executable_directory(), 'index', 'index_words.csv')
    if not os.path.exists(filepath):
        log.error(f"File not found: {filepath}")
        return [], []

    wordList = []
    codeList = []
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 逐行读取数据
        for row in reader:
            codes = set()
            for code in row[1].split(','):
                codes.add(code.strip())
            wordList.append(row[0])
            codeList.append(codes)
    return wordList, codeList

def get_dict_words_last_modify_time() -> str:
    filepath = os.path.join(basic.func.get_executable_directory(), 'dict', 'dict_words.csv')
    return basic.func.get_file_last_modify_time(filepath)

def get_index_words_last_modify_time() -> str:
    filepath = os.path.join(basic.func.get_executable_directory(), 'index', 'index_words.csv')
    return basic.func.get_file_last_modify_time(filepath)