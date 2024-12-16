from .graph_state import GraphState
import pickle
from langchain_community.vectorstores import FAISS
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

from collections import defaultdict
from .prompts.Prompt import get_prompt

from .embedding_model.load_model import get_bge_embeddingModel
embeddings = get_bge_embeddingModel()


# =========================== SETTING VALUES ========================================
import os
SYSTEM_PROMT, USER_PROMPT = get_prompt()
ENSEMBLE_WEIGHTS = [0.7,0.3]
OS_EMBED_PATH = "./embed_top1/" # 800 CHUNK, 100 OVER


cache_retriever = defaultdict(list)
# =========================== FUNCTION SECTION ======================================
def make_retriever(pairs : list, top_k : int) -> EnsembleRetriever:
    docs = [Document(page_content=sentence) for sentence, _ in pairs]

    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    kiwi = KiwiBM25Retriever.from_texts([sentence for sentence, _ in pairs])
    kiwi.k = top_k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[kiwi, retriever],  
        weights=ENSEMBLE_WEIGHTS,  
        search_type='mmr',
    ).with_config({"run_name": "ensemble_retriever"})

    return ensemble_retriever

# =========================== NODE SECTION ======================================
def load_retriever(state: GraphState) -> GraphState:
    top_k = state['retriever_k']
    pdf_path = state['pdf_path']
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_path = os.path.join(current_dir, OS_EMBED_PATH + pdf_path)

    table_pkl_path = current_path + ".pdf_table.pkl"
    markdown_pkl_path = current_path + ".pdf_markdown.pkl"

    if table_pkl_path not in cache_retriever:
        with open(table_pkl_path, 'rb') as file:
            table_pairs = pickle.load(file)

        table_retriever = make_retriever(table_pairs, top_k)
        cache_retriever[table_pkl_path].append(table_retriever)
    else:
        table_retriever = cache_retriever[table_pkl_path][0]

    if markdown_pkl_path not in cache_retriever:   
        with open(markdown_pkl_path,'rb') as file:
            markdown_pairs = pickle.load(file)

        markdown_retriever = make_retriever(markdown_pairs, top_k)
        cache_retriever[markdown_pkl_path].append(markdown_retriever)
    else:
        markdown_retriever = cache_retriever[markdown_pkl_path][0]        
      
    return GraphState(
        table_retriever = table_retriever,
        markdown_retriever = markdown_retriever,
    )    
    
def retrieval(state: GraphState) -> GraphState:

    question = state['question']
    
    table_retriever = state['table_retriever']
    markdown_retriever = state['markdown_retriever']

    table_format = ""
    table = table_retriever.invoke(question)

    for doc in table:
        table_format += doc.page_content+"\n"
          
    markdown_format = ""
    markdown = markdown_retriever.invoke(question)

    for doc in markdown:
        markdown_format += doc.page_content+"\n"
    

    return GraphState(
        original_question = question,
        question = USER_PROMPT.format(
            markdown=markdown_format,
            table= table_format,
            user_input = question
        ),
        context_table = table_format,
        context_text = markdown_format,
    )