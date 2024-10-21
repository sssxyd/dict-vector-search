from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
model.save('model/distiluse-base-multilingual-cased-v1')

