from pathlib import Path
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def get_pipeline(task, model, tokenizer, max_new_tokens=2048, do_sample=False, return_full_text=False):
    return pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_full_text=return_full_text,
    )


# Initializes the LLM model
def crea_llm(model_name: str) -> ChatHuggingFace:
    """
    Loads the given model locally and exposes it as a LangChain ChatModel,
    so prompt templates with messages (system/human) can be used and the
    model's own chat template is applied.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    model.generation_config.max_length = None

    hf_pipeline = get_pipeline("text-generation", model, tokenizer)

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return ChatHuggingFace(llm=llm)

DB_PATH = Path(__file__).resolve().parent.parent / "databases" / "negozio.db"
DB = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")