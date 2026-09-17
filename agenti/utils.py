from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def get_pipeline(task, model, tokenizer, max_new_tokens = 128, do_sample = False, return_full_text = False):

    return pipeline \
    (
        task = task, 
        model = model,
        tokenizer = tokenizer,
        max_new_tokens = max_new_tokens,
        do_sample = do_sample,
        return_full_text = return_full_text
    )

# Inizializza il modello LLM
def crea_llm(model_name: str) -> ChatHuggingFace:
    """
    It loads the model passed as a parameter locally and exposes it as a LangChain `ChatModel`, 
    enabling the use of message-based prompt templates (system/human) and leveraging the model's chat capabilities.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    hf_pipeline = get_pipeline("text-generation", model, tokenizer)

    llm = HuggingFacePipeline(pipeline = hf_pipeline)
    return ChatHuggingFace(llm = llm)

# Database connection, shared among all agents.
DB_PATH = "negozio.db" # Fictitious Database
DB = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

def get_descrizione_dominio() -> str:
    """
    A brief textual description of the database domain (table names only, not the full schema), 
    useful for Agent 1 to determine if a request is relevant, without the full schema details—which are
    needed only by Agent 2 to generate the actual SQL query.
    """
    tabelle = DB.get_usable_table_names()
    return f"Il database contiene le seguenti tabelle: {', '.join(tabelle)}."