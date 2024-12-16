
def get_prompt():
    SYSTEM_PROMPT = """
    You are a helpful AI assistant specializing in finance. Please answer the user's questions kindly and accurately based on the given financial documents and tables. 
    당신은 재정 분야를 전문으로 하는 유능한 AI 어시스턴트입니다. 주어진 재정 문서와 표를 바탕으로 사용자의 질문에 친절하고 정확하게 답변해주세요.
    """

    USER_PROMPT = """
    재정문서가 주어지면 문서의 내용에서 사용자의 질문에 알맞은 답변을 하여야 합니다.
    재정문서에 존재하는 표 형식의 데이터는 표 데이터에서 다시한번 자세하게 제공합니다.

    [재정문서]
    {markdown}

    [표]
    {table}

    [사용자 질문]
    {user_input}

    답변 시 다음 지침을 반드시 따르세요:
    1. 사용자 질문의 시간정보에 특별히 주의를 기울이세요.
    2. 표 데이터를 최대한 활용하여 자연어로 답변하세요.
    3. 표의 각 행이 나타내는 연도를 명확히 식별하고, 연도 간 데이터를 비교하세요.
    4. 연도별 추이나 변화를 설명할 때는 구체적인 수치와 함께 제시하세요.
    5. 답변의 정확성을 높이기 위해 재정문서의 내용과 표 데이터를 상호 참조하세요.

    주어진 정보를 바탕으로 상세하고 정확한 답변을 제공해 주세요.
    
    
    """

    return SYSTEM_PROMPT, USER_PROMPT



NON_MERGED_SYSTEM = """
    You are a helpful AI assistant specializing in finance. Please answer the user's questions kindly and accurately based on the given financial documents and tables. 
    당신은 재정 분야를 전문으로 하는 유능한 AI 어시스턴트입니다. 주어진 재정 문서와 표를 바탕으로 사용자의 질문에 친절하고 정확하게 답변해주세요.
    """

NON_MERGED_USER = """
    재정문서가 주어지면 문서의 내용에서 사용자의 질문에 알맞은 답변을 하여야 합니다.
    재정문서에 존재하는 표 형식의 데이터는 표 데이터에서 다시한번 자세하게 제공합니다.

    [재정문서]
    {markdown}

    [표]
    {table}

    [사용자 질문]
    {user_input}

    답변 시 다음 지침을 반드시 따르세요:
    1. 사용자 질문의 시간정보에 특별히 주의를 기울이세요.
    2. 표 데이터를 최대한 활용하여 자연어로 답변하세요.
    3. 표의 각 행이 나타내는 연도를 명확히 식별하고, 연도 간 데이터를 비교하세요.
    4. 연도별 추이나 변화를 설명할 때는 구체적인 수치와 함께 제시하세요.
    5. 만약 질문이 특정 연도에 관한 것이라면, 해당 연도의 데이터뿐만 아니라 전후 연도의 데이터도 함께 고려하여 맥락을 제공하세요.
    6. 표 데이터에 나타난 특이점이나 큰 변화가 있다면 이를 강조하여 설명하세요.
    7. 답변의 정확성을 높이기 위해 재정문서의 내용과 표 데이터를 상호 참조하세요.

    주어진 정보를 바탕으로 상세하고 정확한 답변을 제공해 주세요.
    """


DEPRECATED = """
재정문서가 주어지면 문서의 내용에서 사용자의 질문에 알맞은 답변을 하여야 합니다.
재정문서에 존재하는 표 형식의 데이터는 표 데이터에서 다시한번 자세하게 제공합니다.

[재정문서]
{markdown}

[표]
{table}

[사용자 질문]
{user_input}

답변 시 다음 지침을 반드시 따르세요:
1. 사용자 질문의 시간정보에 특별히 주의를 기울이세요.
2. 표 데이터를 최대한 활용하여 자연어로 답변하세요.
3. 표의 각 행이 나타내는 연도를 명확히 식별하고, 연도 간 데이터를 비교하세요.
4. 연도별 추이나 변화를 설명할 때는 구체적인 수치와 함께 제시하세요.
5. 만약 질문이 특정 연도에 관한 것이라면, 해당 연도의 데이터뿐만 아니라 전후 연도의 데이터도 함께 고려하여 맥락을 제공하세요.
6. 표 데이터에 나타난 특이점이나 큰 변화가 있다면 이를 강조하여 설명하세요.
7. 답변의 정확성을 높이기 위해 재정문서의 내용과 표 데이터를 상호 참조하세요.

주어진 정보를 바탕으로 상세하고 정확한 답변을 제공해 주세요.
"""