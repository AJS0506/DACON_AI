from .graph_state import GraphState
from .models.llm_load import get_FullFineTuningModel
from .prompts.Prompt import get_prompt

SYSTEM_PROMPT, USER_PROMPT = get_prompt()
pipeline = get_FullFineTuningModel()

def llm_answer(state: GraphState) -> GraphState:
    global SYSTEM_PROMPT

    messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}"},
            {"role": "user", "content": f"{state['question']}"}
        ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(
        prompt, 
        max_new_tokens=1024, 
        eos_token_id=terminators, 
        do_sample=False,
    )

    generated_text = outputs[0]["generated_text"][len(prompt):]
    
    return GraphState(
        answer = generated_text
    )