import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import PGEmbedding
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv("./.env")
modelName = os.getenv("LLM_MODEL", "llama3")
ollamaUrl = os.getenv("OLLAMA_URL")
collName = os.getenv("RULE_COLL_NAME", "rule-docs")
vectorDBUrl = os.getenv("VECTOR_DB_URL")

def connectToDB(modelName, ollamaUrl, vectorDBUrl):
    embeddingFunction = OllamaEmbeddings(model=modelName, base_url=ollamaUrl)
    db = PGEmbedding(
        connection_string=vectorDBUrl,
        embedding_function=embeddingFunction,
        collection_name=collName
    )
    return db

db = connectToDB(modelName, ollamaUrl, vectorDBUrl)

search_res = db.similarity_search("Ken Cunningham", k=5)
print(search_res)
