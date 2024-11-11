import os
import argparse
from dotenv import load_dotenv
from typing import List, Tuple

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGEmbedding
from sqlalchemy import create_engine

load_dotenv("./.env")

def loadDocuments(directory):
    print("Loading documents")
    loader = DirectoryLoader(directory)
    documents = loader.load()
    print("Documents loaded")
    print("Splitting documents")
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    docs = textSplitter.split_documents(documents)
    for doc in docs:
        doc.page_content = f"search_document: {doc.page_content}"
    print("Documents split")
    return docs

def storeVectors(modelName, ollamaUrl, vectorDBUrl, collname, docs):

    embeddingFunction = OllamaEmbeddings(model=modelName, base_url=ollamaUrl)
    print("Embedding function loaded")
    print("Beginning embedding")
    db = PGEmbedding.from_documents(
        embedding=embeddingFunction,
        documents=docs,
        collection_name=collname,
        connection_string=vectorDBUrl,
        pre_delete_collection=True
    )
    print("Embedding complete")

    return db

def connectToDB(modelName, ollamaUrl, vectorDBUrl):
    embeddingFunction = OllamaEmbeddings(model=modelName, base_url=ollamaUrl)
    db = PGEmbedding(
        connection_string=vectorDBUrl,
        embedding_function=embeddingFunction
    )
    return db

def createIndex(db):
    db.create_hnsw_index(max_elements=10000, dims=1536, m=8, ef_construction=16, ef_search=16)

def main():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--model", type=str, default=os.getenv("EMBED_MODEL"), help="Model to use for embeddings")
    parser.add_argument("--ollama_url", type=str, default=os.getenv("OLLAMA_URL"), help="URL to the ollama model")
    parser.add_argument("--directory", type=str, default="./rule-docs", help="Directory containing files to create embedding for")
    parser.add_argument("--collname", type=str, default=os.getenv("RULE_COLL_NAME"), help="collection name to store the embeddings")
    parser.add_argument("--vector_url", type=str, default=os.getenv("VECTOR_DB_URL"), help="URL to the pg_embedding vector database")

    args = parser.parse_args()
    modelName=args.model
    ollamaUrl=args.ollama_url
    rag_dir=args.directory
    vectorDBUrl = args.vector_url
    collname=args.collname

    docs = loadDocuments(rag_dir)

    storeVectors(modelName, ollamaUrl, vectorDBUrl, collname, docs)
    # docsWScore: List[Tuple[Document, float]] = db.similarity_search_with_score("Who is ann whitt")
    # for doc, score in docsWScore:
    #     print("-" * 80)
    #     print("Score: ", score)
    #     print(doc.page_content)
    #     print("-" * 80)

if __name__ == "__main__":
    main()
