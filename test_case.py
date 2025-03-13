import numpy as np

from service import dictWords
from service import aiModel
from service import vectorIndex

word = '左旋维c油'
model = aiModel.load_sentence_transformer_model()

v1 = model.encode([word])

# print(np.vstack(v1)[0])

dwords = dictWords.load_dict_word_set()
words, codes =dictWords.load_index_word_codes()

word_index, pinyin_index = vectorIndex.load_vector_indexes()

k = 6
distances, indices = word_index.search(v1, k)
for i in range(k):
    word_index = indices[0][i]
    similar_codes = codes[word_index]
    distance = distances[0][i]
    similar_word = words[word_index]
    real_words = []
    for code in similar_codes:
        real_words.append(dwords[code].word)
    print("{}, {:.4f}, {}".format(similar_word, distance, '|'.join(real_words)))
print("--------------------")

py = dictWords.pinyin_word(word)
print(py)
v2 = model.encode([py])
# print(np.vstack(v2)[0])
distances, indices = pinyin_index.search(v2, k)
for i in range(k):
    word_index = indices[0][i]
    similar_codes = codes[word_index]
    distance = distances[0][i]
    similar_word = words[word_index]
    real_words = []
    for code in similar_codes:
        real_words.append(dwords[code].word)
    print("{}, {:.4f}, {}".format(similar_word, distance, '|'.join(real_words)))


vectorIndex.search_vector_indexes()