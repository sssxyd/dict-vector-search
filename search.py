import multiprocessing
from datetime import datetime

import basic
from service import dictWords
from service import vectorIndex
from service import aiModel

# dict_words = dict.collect_dict_words(split=True)
# dict.save_dict_words(dict_words)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 对于 Windows 上的可执行文件打包是必要的
    model = aiModel.load_sentence_transformer_model()
    word_index, pinyin_index = vectorIndex.load_vector_indexes()
    words = dictWords.load_dict_word_set()
    _, codes = dictWords.load_index_word_codes()

    search_words = ["太尤双黄连口服液", "复访999感冒灵", "反应亭", "反映亭", "双瓜唐安", "丁贵儿骑贴", "VE三油胶丸", "氯已定",
                    "六五四二胃药", "475被他", "她达拉非", "太极降脂宁"]
    for search_word in search_words:
        start = datetime.now()
        results = vectorIndex.search_vector_indexes(word=search_word, model=model, word_index=word_index,
                                                    pinyin_index=pinyin_index, index_codes=codes, dict_words=words, top_k=5)
        print(f"-------search '{search_word}' cost {basic.func.get_duration(start)}--------")
        for dict_word in results:
            print(dict_word)
        print("-------------------------------------------------")