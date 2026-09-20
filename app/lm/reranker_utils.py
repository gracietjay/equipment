from FlagEmbedding import FlagReranker
from app.conf.reranker_config import reranker_config

_reranker_model = None

def get_reranker_model():
    global _reranker_model  
    if _reranker_model is None:
        _reranker_model= FlagReranker(
            model_name_or_path=reranker_config.bge_reranker_large,
            device=reranker_config.bge_reranker_device,
            use_fp16=reranker_config.bge_reranker_fp16
        )
    return _reranker_model