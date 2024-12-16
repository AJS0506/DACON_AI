
from typing import Annotated, Sequence, TypedDict
from faiss import IndexFlatL2  

class GraphState(TypedDict):
    # PDF 경로 
    pdf_path: Annotated[str, "Pdf_path (to make Retriever)"]

    # 질문-답변 
    question: Annotated[str, "User Question"]
    original_question: Annotated[str, 'Caching Question.']
    answer: Annotated[str, "LLM answer"]
    relevant : Annotated[str, "Relevant Check"]
    relevant_summary : Annotated[str,"Relevant에 대한 근거"]

    # 임베딩값 및 검색결과 
    # embed_pair: Annotated[List[List[str]], "[원본문장, 임베딩] 2d list"]
    context_text: Annotated[Sequence[str], "markdown retrieval results"]
    context_table: Annotated[Sequence[str], "Table retrieval results"]

    # 검색기 관련 
    retriever_k: Annotated[int, "Top-K retrieval"]
    table_retriever: Annotated[IndexFlatL2, "Retriever Object"]
    markdown_retriever: Annotated[IndexFlatL2, "Retriever Object"]