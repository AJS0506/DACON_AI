from .graph_state import GraphState

def start_of_node(state: GraphState) -> GraphState:
    # 초기 작업 수행 -> 현재 아무것도 없음
    
    return GraphState(
        retriever_k = state['retriever_k'],
        question = state['question'],
        pdf_path = state['pdf_path'],
    )