from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama 
from langchain_openai import ChatOpenAI
from src.config.settings import Settings
from dotenv import load_dotenv

load_dotenv()

def get_llm(provider, model_choice, api_key, api_url = ""):
    # Initialize the LLM based on user selection
    if provider == "OpenAI":
        llm = ChatOpenAI(model=model_choice, api_key=api_key)

    elif provider == "Groq":
        llm = ChatGroq(model=model_choice, api_key=api_key)

    elif provider == "Ollama (Local)":
        # Connects directly to their local machine
        llm = ChatOllama(model=model_choice, base_url=api_url)

    # llm = ChatOllama(
    #     model= "qwen2.5:7b",
    #     temperature= 0.0,
    #     base_url = "https://carey-dissatisfied-disingenuously.ngrok-free.dev"
    # )
    return llm


# from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama

