import transformers, torch, warnings
from peft import PeftModel, PeftConfig
import os

# Top_p, dosample 워닝 무시 
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")

# ======================== LLM LOAD 구현 ==============================
def get_FullFineTuningModel():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "llama3.1_RAGTUNE_3epoch_3RAG_800|100_1e5|markdown|tabula")
    
    pipeline = transformers.pipeline(
        "text-generation",
        model = model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda:3",
    )

    pipeline.model.eval()

    return pipeline

def get_LoRAMergedModel():
    config = PeftConfig.from_pretrained("./llama3.1_70b_5epoch_r16")
    
    # 기본 모델 로드
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # LoRA 가중치 적용
    model = PeftModel.from_pretrained(model, "./llama3.1_70b_5epoch_r16")

    # LORA 로드 확인
    print(model)
    print(model.peft_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
    
    # 파이프라인 생성
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    pipeline.model.eval()
    
    return pipeline