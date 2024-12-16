from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image
import io
from PIL import Image as PILImage

from tqdm import tqdm

from nodes import start_of_node, load_retriever, retrieval, llm_answer, load_retriever, relevant_check, rewrite, is_relevant, GraphState
from load_testdata.make_pair import get_questions
from typing import Annotated, List, Tuple

# [PDF_PATH, QUESTION, ANSWER]
questionInfo = Annotated[
    Tuple[str, str, str],
    "PDF_PATH", "QUESTION", "ANSWER"
]
questions: List[questionInfo] = get_questions()

# DEFINITION : nodes -> graph_state.py
financial_flow = StateGraph(GraphState)

# 노드 추가
financial_flow.add_node("start_of_node",start_of_node)
financial_flow.add_node("load_retriever",load_retriever)
financial_flow.add_node("retrieval",retrieval)
#financial_flow.add_node("relevant_check",relevant_check)
#financial_flow.add_node('rewrite',rewrite)
financial_flow.add_node("llm_answer",llm_answer)

# 엣지 추가
financial_flow.add_edge("start_of_node", "load_retriever")
financial_flow.add_edge("load_retriever", "retrieval")
financial_flow.add_edge("retrieval", "llm_answer")
financial_flow.add_edge("llm_answer",END)
#financial_flow.add_edge("rewrite","retrieval")
#financial_flow.add_edge("relevant_check", END)

# 조건부 엣지 추가 
# financial_flow.add_conditional_edges(
#     "relevant_check",is_relevant,
#     {
#         "TRUE" : END,
#         #"FALSE" : "rewrite",
#         "FALSE" : END,
#     }
# )

# 그래프 시작 지점 설정
financial_flow.set_entry_point("start_of_node")
#memory = MemorySaver() -> 이거 하니 그래프 상태 날아감;
graph = financial_flow.compile()#(checkpointer=memory)


# 그래프 시각화
test_result = Image(graph.get_graph(xray=True).draw_mermaid_png())
image_data = test_result.data
pil_image = PILImage.open(io.BytesIO(image_data))
pil_image.save('GraphFlow.png', 'PNG')

# 그래프 입력 설정
from langchain_core.runnables import RunnableConfig
config = RunnableConfig(recursion_limit=60,configurable={"thread_id":"FINANCIAL_RAG"})

questions = get_questions()

fp = open("./results/sample_submission.csv",'r',encoding='utf8')
fp2 = open("./results/result.csv",'w',encoding='utf8')
fp2.write(fp.readline())

for question in tqdm(questions):
    file_path = question[1]
    query = question[-1]

    row = fp.readline().strip().split(',')

    print("=====질문=====")
    print(query)

    input_state = GraphState(
        question = query, # 질문 입력
        pdf_path = file_path, # 어떤 PDF PATH로부터 Retriever 
        retriever_k = 4, # 주입하는 Context 몇개 (6의 경우 markdown 6 + table 6 = 12)
    )

    output = graph.invoke(input_state, config)
    answer = output['answer']

    print("=====답변=====")
    print(answer)

    answer = answer.replace("\n"," ").replace(",","")
    row[-1] = answer

    output = ",".join(row)+"\n"
    fp2.write(output)


