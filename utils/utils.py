import functools
from pathlib import Path
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
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
@functools.lru_cache(maxsize=None)
def crea_llm(model_name: str, carica_in_4bit: bool = False) -> ChatHuggingFace:

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if carica_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={"": 0},
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )

    model.generation_config.max_length = None

    hf_pipeline = get_pipeline(
        "text-generation",
        model,
        tokenizer
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    return ChatHuggingFace(llm=llm)


DB_PATH = Path(__file__).resolve().parent.parent / "databases" / "container_logistics.sqlite"
DB = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")