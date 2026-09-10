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
    Carica il modello inserito come parametro in locale e lo espone come ChatModel
    LangChain, cosi' da poter usare i prompt template a messaggi
    (system/human) e sfruttare il chat template del modello.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    hf_pipeline = get_pipeline("text-generation", model, tokenizer)

    llm = HuggingFacePipeline(pipeline = hf_pipeline)
    return ChatHuggingFace(llm = llm)

# Connessione al database, condivisa tra tutti gli agenti.
DB_PATH = "negozio.db" # Database Fittizio
DB = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

def get_descrizione_dominio() -> str:
    """
    Breve descrizione testuale del dominio del database (solo nomi delle
    tabelle, non lo schema completo), utile all'Agente 1 per capire se una
    richiesta e' pertinente, senza il dettagli sullo schema completo
    che invece serve solo all'Agente 2 per generare la query SQL vera e propria.
    """
    tabelle = DB.get_usable_table_names()
    return f"The database contains the following tables: {', '.join(tabelle)}."