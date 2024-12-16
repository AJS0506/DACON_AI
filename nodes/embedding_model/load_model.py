from langchain_huggingface import HuggingFaceEmbeddings

def get_bge_embeddingModel():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",model_kwargs={'device':'cuda'})
    return embeddings

def get_multilangul_embeddingModel():
    pass

def get_nvidia_embeddingModel():
    pass

def get_solar_embeddingModel():
    pass