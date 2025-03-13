import os.path

from sentence_transformers import SentenceTransformer

import basic

def load_sentence_transformer_model() -> SentenceTransformer | None:
    model_path = os.path.join(basic.func.get_executable_directory(), 'model', 'distiluse-base-multilingual-cased-v1')
    if not os.path.exists(model_path):
        return None
    # 显式指定设备为CPU
    model = SentenceTransformer(
        model_name_or_path=model_path,
        device='cpu'  # 强制在CPU上加载
    )
    print(f"loaded sentence transformer model from {model_path}")
    return model.eval().to('cpu')  # 双保险确保设备位置