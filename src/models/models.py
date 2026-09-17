from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def embedding_model(embed_provider,embed_api_key,embed_model_choice,embed_api_url):

    # Initialize the Embedding model based on user selection
    if embed_provider == "OpenAI":
        embeddings = OpenAIEmbeddings(
            model=embed_model_choice, 
            api_key=embed_api_key
        )

    elif embed_provider == "HuggingFace (Local)":
        # Downloads the model to the user's machine on first run
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_choice
        )

    elif embed_provider == "Ollama (Local)":
        # Calls the local Ollama server
        embeddings = OllamaEmbeddings(
            model=embed_model_choice, 
            base_url=embed_api_url
        )
    # model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    return embeddings

def embd_model_name(embed_provider):
    if embed_provider == "OpenAI":
        model = embedding_model()

    elif embed_provider == "HuggingFace (Local)":
        model = embedding_model()

    elif embed_provider == "Ollama (Local)":
        model = embedding_model()
    
    return model.model_name