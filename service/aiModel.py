import os.path

from sentence_transformers import SentenceTransformer

import basic

def load_sentence_transformer_model() -> SentenceTransformer:
    log = basic.log()
    model_path = os.path.join(basic.func.get_executable_directory(), 'model', 'distiluse-base-multilingual-cased-v1')
    model = SentenceTransformer(model_name_or_path=model_path)
    log.info(f">>loaded sentence transformer model from {model_path}")
    return model

