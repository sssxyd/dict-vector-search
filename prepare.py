import multiprocessing
from datetime import datetime

import basic.func
from service import dictWords
from service import aiModel
from service import vectorIndex

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 对于 Windows 上的可执行文件打包是必要的
    start_time = datetime.now()
    # 收集txt，生成搜索词字典
    dictWords.prepare_dict_words()

    # 根据字典，生成索引词词条ID的关系
    # dictWords.prepare_index_words(3, 5)

    # 生成向量索引
    # words, _ = dictWords.load_index_word_codes()
    # model = aiModel.load_sentence_transformer_model()
    # vectorIndex.create_vector_indexes(words, model)

    print(f"cost {basic.func.get_duration(start_time)}")