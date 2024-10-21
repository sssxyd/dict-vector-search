import multiprocessing
import os.path
from datetime import datetime

import basic.func
from service import dictWords
from service import aiModel
from service import vectorIndex

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 对于 Windows 上的可执行文件打包是必要的
    start_time = datetime.now()
    # 收集txt，生成搜索词字典
    # dictWords.prepare_dict_words()

    # 根据字典，生成索引词词条ID的关系
    batch_index_dir = os.path.join(basic.func.get_executable_directory(), 'index', start_time.strftime("%Y%m%d%H%M%S"))
    words = dictWords.prepare_index_words(batch_index_dir, 3, 5)

    # 生成向量索引
    model = aiModel.load_sentence_transformer_model()
    vectorIndex.create_vector_indexes(batch_index_dir, words, model)

    print(f"cost {basic.func.get_duration(start_time)}")