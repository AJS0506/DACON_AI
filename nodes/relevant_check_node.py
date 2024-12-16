from .graph_state import GraphState
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

check_prompt = """
당신은 재정 AI 교수 입니다.
[문제]와 학생이 작성한 [답변]이 주어집니다.

문제는 한개 또는 두개 이상을 물어볼 수 있습니다.
학생이 답한 내용이 문제의 의도에 부합하는지 판단하여야 합니다.

판단을 할때 다음 유의사항을 따르세요.
1. 두개이상의 질문에 한개의 답변을 하면 안됩니다.
2. 필요없는 부가정보를 답변하면 안됩니다.
3. 답변 내용의 정확성이 아닌, 의도를 판단하세요.
4. 부정확한 답변이어도 문제의 의도에 맞는 답변이면 TRUE 입니다. 

[문제]
{question}

[답변]
{answer}

문제의 의도에 부합하면 TRUE만 답변하며
그렇지않으면 사유와 함께 FALSE 로 답변합니다.
"""


prompt = PromptTemplate.from_template(check_prompt)
llm = ChatOllama(model='llama3.1:70b-instruct-q8_0', temperature=0.4)
parser = StrOutputParser()
chain = prompt | llm | parser


def relevant_check(state:GraphState) -> GraphState:
    global chain
    
    question = state['original_question']
    answer = state['answer']
    
    relevance_answer = chain.invoke(
        {
         'question':question,
         'answer':answer
        }
    )
    
    is_relevance = None
    if "TRUE" in relevance_answer:
        is_relevance = "TRUE"
    else:
        is_relevance = "FALSE"    
    
    if is_relevance == "FALSE":
        print("유효성 검증 실패. 사유 -> ",relevance_answer)
    
    return GraphState(
        relevant = is_relevance,
        relevant_summary = relevance_answer
    )

# ================= 조건부 엣지 라우팅 함수 구현 =================
def is_relevant(state:GraphState) -> str:
    if state['relevant'] == "TRUE":
        return "TRUE"
    else:
        return "FALSE"
# ==============================================================

from langchain_core.prompts import ChatPromptTemplate

def rewrite(state : GraphState) -> GraphState:
    # rewrite_prompt = """
    # 당신은 유능한 답변 재작성자 AI어시스턴트 입니다.
    
    # [질문]과 잘못된 [답변]이 주어지면, 질문을 더 구체화하여 재작성합니다.
    # 답변에서 부족한 부분이 [유의사항]으로 주어집니다.
    
    # [질문]
    # {question}
    
    # [답변]
    # {answer}
    
    # [유의사항]
    # {relevant_summary}
    
    # [답변] 과 [유의사항]을 참고하여 [질문]을 새롭게 생성합니다.
    # 원본 질문의 의미를 훼손하면 안되며 단어 및 문장형태의 변형만 하여야 합니다.
    # """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional prompt rewriter. Your task is to improve the question. Question must be written in korean language. Don't narrate, just reponse an improved question.",
            ),
            (
                "human",
                "Look at the input and try to reason about the underlying semantic intent / meaning."
                "\n\nHere is the initial question:\n ------- \n{question}\n ------- \n"
                "\n\nFormulate an improved and short question:",
            ),
        ]
    )
    question = state['original_question']
    answer = state['answer']
    relevant_summary = state['relevant_summary']
    
    #prompt = PromptTemplate.from_template(rewrite_prompt)
    chain = prompt | llm | parser
    
    rewrited_question = chain.invoke(
        {
            "question" : question,
            #"answer" : answer,
            #"relevant_summary" : relevant_summary
        }
    )
    print("질문을 재작성합니다.")
    print("재작성된 질문 -> ",rewrited_question)
    return GraphState(
        question = rewrited_question
    )
    
