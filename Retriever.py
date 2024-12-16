import pickle
from langchain_community.vectorstores import FAISS
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",model_kwargs={'device':'cuda'})


# 입력값 paris = [원본문장, 임베딩행렬]
def make_retriever(pairs : list) -> FAISS:
    top_k = 3

    #FAISS 벡터스토어를 사용하기 위해 Document 객체로 변환
    docs = [Document(page_content=sentence) for sentence, _ in pairs]

    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    kiwi = KiwiBM25Retriever.from_texts([sentence for sentence, _ in pairs])
    kiwi.k = top_k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[kiwi, retriever],  
        weights=[0.7,0.3],  
        search_type='mmr',
    ).with_config({"run_name": "ensemble_retriever"})

    return ensemble_retriever

from collections import defaultdict
cache_retriever = defaultdict(list)

def get_marktable_relevant(pdf_path:str, query:str):
    table_pkl_path = "../embed_top1/"+pdf_path+".pdf_table.pkl"
    markdown_pkl_path = "../embed_top1/"+pdf_path+".pdf_markdown.pkl"

    if table_pkl_path not in cache_retriever:
        with open(table_pkl_path, 'rb') as file:
            table_pairs = pickle.load(file)

        table_retriever = make_retriever(table_pairs)
        cache_retriever[table_pkl_path].append(table_retriever)
    else:
        table_retriever = cache_retriever[table_pkl_path][0]

    if markdown_pkl_path not in cache_retriever:   
        with open(markdown_pkl_path,'rb') as file:
            markdown_pairs = pickle.load(file)

        markdown_retriever = make_retriever(markdown_pairs)
        cache_retriever[markdown_pkl_path].append(markdown_retriever)
    else:
        markdown_retriever = cache_retriever[markdown_pkl_path][0]        

    # 테이블, 마크다운 각각 리트리버 검색 수행
    tables = table_retriever.invoke(query)
    markdowns = markdown_retriever.invoke(query)

    # Document 객체 결과를 page_content 형태로 변환
    tables = [doc.page_content for doc in tables]
    markdowns = [doc.page_content for doc in markdowns]

    return tables, markdowns

# FAIL
# def merged_marktable_relevant(pdf_path:str, query:str):
#     pkl_path = "../embed/"+pdf_path+".pdf_merged.pkl"

#     if pkl_path not in cache_retriever:
#         with open(pkl_path,'rb') as file:
#             merged_pairs = pickle.load(file)

#         merged_retriever = make_retriever(merged_pairs)
#         cache_retriever[pkl_path] = merged_retriever
#     else:
#         merged_retriever = cache_retriever[pkl_path]

#     result = merged_retriever.invoke(query)
#     result = [doc.page_content for doc in result]

#     return result    
             