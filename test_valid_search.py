import csv
import multiprocessing
import re
from datetime import datetime

import basic
from service import dictWords
from service import vectorIndex
from service import aiModel
from service.vectorIndex import IndexWord


# dict_words = dict.collect_dict_words(split=True)
# dict.save_dict_words(dict_words)
def read_keywords() -> list[str]:
    with open("validate_keywords.txt", "r", encoding="utf-8") as file:
        return file.readlines()

def split_keyword(word : str) -> list[str]:
    # 匹配所有不是中文、英文、数字的字符
    pattern = r"[^\u4e00-\u9fffA-Za-z0-9]"
    # 使用 re.sub 将匹配到的字符替换为空格
    return re.sub(pattern, " ", word).strip().split()

def search_keywords(key_words : list[str]) -> dict[str, IndexWord]:
    model = aiModel.load_sentence_transformer_model()

    word_index, pinyin_index = vectorIndex.load_vector_indexes()
    words = dictWords.load_dict_word_set()
    _, codes = dictWords.load_index_word_codes()
    print("loaded indexes")
    total = len(key_words)
    count = 1
    similar_set = dict()
    for key_word in key_words:
        key_word = key_word.strip()
        sub_words = split_keyword(key_word)
        item_results = []
        for sub_word in sub_words:
            results = vectorIndex.search_vector_indexes(word=sub_word, model=model, word_index=word_index,
                                                        pinyin_index=pinyin_index, index_codes=codes, dict_words=words,
                                                        top_k=3, pinyin=True)
            item_results.extend(results)
        item_sorted = sorted(item_results, key=lambda x: (-x.score, -x.distance, len(x.word)))
        similar_word = item_sorted[0] if len(item_sorted) > 0 else IndexWord(index="", code="", word="未找到", score=0, distance=0)
        similar_set[key_word] = similar_word
        print(f"{count}/{total}: {key_word} -> {similar_word}")
        count = count + 1
    return similar_set

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 对于 Windows 上的可执行文件打包是必要的
    keywords = read_keywords()
    print(f"read {len(keywords)} keywords")
    search_results = search_keywords(keywords)
    print(f"searched {len(search_results)} keywords")
    filepath = "validate_results.csv"
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 逐行写入数据
        for keyword, index_word in search_results.items():
            writer.writerow([keyword, index_word.word, index_word.score, index_word.distance])
    print(f"saved search results to {filepath}")