import math
import multiprocessing
import os
import re

import faiss
import numpy as np
import psutil
from faiss import IndexFlatL2
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import basic
from .dictWords import DictWord, trim_word, pinyin_word, get_latest_directory


class IndexWord(BaseModel):
    index: str
    code: str
    word: str
    score: int
    distance: float

    def isCredible(self) -> bool:
        if self.index == "PINYIN":
            return True
        if self.score < 1:
            return self.distance < 0.4
        elif self.score == 1:
            if len(self.word) > 3:
                return self.distance < 0.3
            else:
                return self.distance < 0.5
        else:
            return True


def _vector_words_with_model(process_index : int, index_words: list[str], model: SentenceTransformer, batch_size : int = 500,) -> (list[np.ndarray], list[np.ndarray]):
    word_embeddings = []
    pinyin_embeddings = []
    count = 0
    for i in range(0, len(index_words), batch_size):
        batch_words = index_words[i:i + batch_size]
        original_words = []
        pinyin_words = []
        for index_word in batch_words:
            original_words.append(index_word)
            pinyin_words.append(pinyin_word(index_word))
        count += len(original_words)
        word_embeddings.append(model.encode(original_words))
        pinyin_embeddings.append(model.encode(pinyin_words))
        print(f"Process[{process_index}] Indexing {count}/{len(index_words)} index words")
    word_embeddings = np.vstack(word_embeddings)
    pinyin_embeddings = np.vstack(pinyin_embeddings)
    return word_embeddings, pinyin_embeddings


def create_vector_indexes(batch_index_dir : str, index_words : list[str], model : SentenceTransformer, worker : int = 0, batch_size : int = 500):
    word_embeddings = []
    pinyin_embeddings = []
    if worker == 0:
        worker = psutil.cpu_count(logical=True)
    worker_size = math.ceil(len(index_words)/worker)
    print(f"Creating indexes with {worker} workers, each worker process {worker_size} words")
    if worker == 1:
        word_embeddings, pinyin_embeddings = _vector_words_with_model(1, index_words, model, batch_size)
    else:
        with multiprocessing.Pool(processes=worker) as pool:
            worker_index = 1
            results = []
            for i in range(0, len(index_words), worker_size):
                batch_words = index_words[i:i + worker_size]
                result = pool.apply_async(_vector_words_with_model, args=(worker_index, batch_words, model, batch_size))
                results.append(result)
                worker_index = worker_index + 1

            for result in results:
                word_embedding, pinyin_embedding = result.get()
                word_embeddings.append(word_embedding)
                pinyin_embeddings.append(pinyin_embedding)

    word_embeddings = np.vstack(word_embeddings)
    pinyin_embeddings = np.vstack(pinyin_embeddings)
    # 创建FAISS索引
    d = word_embeddings.shape[1]  # 向量维度
    word_index = faiss.IndexFlatL2(d)  # 使用L2距离
    word_index.add(word_embeddings)    # 添加向量到索引
    word_index_file_path = os.path.join(batch_index_dir, 'word_index.bin')
    faiss.write_index(word_index, word_index_file_path)
    print(f"Word index saved to {word_index_file_path}")
    # 创建FAISS索引
    d = pinyin_embeddings.shape[1]  # 向量维度
    pinyin_index = faiss.IndexFlatL2(d)  # 使用L2距离
    pinyin_index.add(pinyin_embeddings)    # 添加向量到索引
    pinyin_index_file_path = os.path.join(batch_index_dir, 'pinyin_index.bin')
    faiss.write_index(pinyin_index, pinyin_index_file_path)
    print(f"Pinyin index saved to {pinyin_index_file_path}")


def load_vector_indexes() -> (IndexFlatL2, IndexFlatL2):
    log = basic.log()  # 确保log函数正确
    batch_index_dir = get_latest_directory()
    if not batch_index_dir:
        log.error("Index directory not found")
        return None, None
    word_index_file_path = os.path.join(batch_index_dir, 'word_index.bin')
    pinyin_index_file_path = os.path.join(batch_index_dir, 'pinyin_index.bin')
    if not os.path.exists(word_index_file_path) or not os.path.exists(pinyin_index_file_path):
        log.error(f"Index file not found: {word_index_file_path} or {pinyin_index_file_path}")
        return None, None
    word_index = faiss.read_index(word_index_file_path)
    pinyin_index = faiss.read_index(pinyin_index_file_path)
    return word_index, pinyin_index


def calculate_match_score(search_word: str, dictionary_word: str) -> int:
    score = 0
    position_dict = {}
    matched_chars = []
    matched_positions = []
    # 创建字典词中每个字符的位置列表
    for idx, char in enumerate(dictionary_word):
        if char not in position_dict:
            position_dict[char] = []
        position_dict[char].append(idx)

    # 遍历搜索词中的每个字符，第一个规则
    for idx, char in enumerate(search_word):
        if char in position_dict:
            score += 1  # 每个字符出现得1分
            matched_chars.append(char)
            matched_positions.append(idx)

    # 第二个和第三个规则：检查连续字符顺序和间隔
    for i in range(len(matched_chars) - 1):
        first_char = matched_chars[i]
        second_char = matched_chars[i + 1]
        first_positions = position_dict[first_char]
        second_positions = position_dict[second_char]
        found_consecutive = False
        for pos1 in first_positions:
            for pos2 in second_positions:
                if pos2 > pos1:
                    if not found_consecutive:
                        score += 1  # 连续字符按顺序出现得1分
                        found_consecutive = True
                    if pos2 - pos1 == matched_positions[i+1] - matched_positions[i]:
                        score += 1  # 匹配字符间隔相同, 得1分
                    break  # 停止检查该字符对的其他位置，因为已找到合适的顺序
            if found_consecutive:
                break

    return score

def _search_vector_indexes(key_word: str, pinyin: bool,  model: SentenceTransformer, vector_index: IndexFlatL2,
                           index_codes: list[set[str]], dict_words: dict[str, DictWord],top_k : int) -> list[IndexWord]:
    word_vector = model.encode([key_word] if not pinyin else [pinyin_word(key_word)])
    distances, indices = vector_index.search(word_vector, top_k)
    results = []
    for i in range(top_k):
        word_index = indices[0][i]
        similar_codes = index_codes[word_index]
        distance = distances[0][i]
        for code in similar_codes:
            similar_word = dict_words[code]
            score = calculate_match_score(key_word, similar_word.word)
            iw = IndexWord(index="PINYIN" if pinyin else "WORD", code=code, word=similar_word.word, score=score, distance=distance)
            if iw.isCredible():
                results.append(iw)
    return results

def get_word_index_last_modify_time() -> str:
    batch_index_dir = get_latest_directory()
    if not batch_index_dir:
        return '1900-01-01 00:00:00'
    filepath = os.path.join(batch_index_dir, 'word_index.bin')
    return basic.func.get_file_last_modify_time(filepath)

def get_pinyin_index_last_modify_time() -> str:
    batch_index_dir = get_latest_directory()
    if not batch_index_dir:
        return '1900-01-01 00:00:00'
    filepath = os.path.join(batch_index_dir, 'pinyin_index.bin')
    return basic.func.get_file_last_modify_time(filepath)

def search_vector_indexes(word: str, model: SentenceTransformer, word_index: IndexFlatL2, pinyin_index : IndexFlatL2,
                          index_codes: list[set[str]], dict_words: dict[str, DictWord], top_k: int = 5, pinyin : bool = False) -> list[IndexWord]:

    key_word = trim_word(word)

    top_n = max(top_k + 5, top_k * 2)

    # 搜索拼音和非拼音的向量索引
    index_words = _search_vector_indexes(key_word, False, model, word_index, index_codes, dict_words, top_n)
    if pinyin:
        index_words += _search_vector_indexes(key_word, True, model, pinyin_index, index_codes, dict_words, top_n)

    # 按照分数和距离排序
    sorted_results = sorted(index_words, key=lambda x: (-x.score, x.distance, len(x.word)))

    # 去除重复的词
    exist_words = set()
    return_index_words = []
    for iw in sorted_results:
        if iw.word in exist_words:
            continue
        exist_words.add(iw.word)
        return_index_words.append(iw)

    return return_index_words[:top_k]