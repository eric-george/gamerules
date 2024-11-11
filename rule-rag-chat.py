import os
import argparse
import json
import uuid

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import PGEmbedding
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

import streamlit as st

load_dotenv("./.env")
modelName = os.getenv("LLM_MODEL", "llama3")
ollamaUrl = os.getenv("OLLAMA_URL")
collName = os.getenv("RULE_COLL_NAME", "rule-docs")
embedModelName = os.getenv("EMBED_MODEL")
vectorDBUrl = os.getenv("VECTOR_DB_URL")

@st.cache_resource
def connectToDB(embedModelName, ollamaUrl, vectorDBUrl, collName):
    embeddingFunction = OllamaEmbeddings(model=embedModelName, base_url=ollamaUrl)
    db = PGEmbedding(
        connection_string=vectorDBUrl,
        embedding_function=embeddingFunction,
        collection_name=collName
    )
    return db

@st.cache_resource
def createModel(modelName, ollamaUrl):
    return ChatOllama(model=modelName, base_url=ollamaUrl, tempurature=0.1)

model = createModel(modelName, ollamaUrl)
db = connectToDB(embedModelName, ollamaUrl, vectorDBUrl, collName)

def query_documents(db, question):
    similar_docs = db.similarity_search(f"search_query: {question}", k=4)
    print(f"SimilarDocs: {len(similar_docs)}")
    return list(map(lambda doc: f"Source: {doc.metadata.get('source', 'NA')}\nContent: {doc.page_content}", similar_docs))

def prompt_ai(db, model, messages):
    # Fetch the relevant documents for the query
    user_prompt = messages[-1].content
    # print(f"User prompt: {user_prompt}")
    retrieved_context = query_documents(db, user_prompt)
    formatted_prompt = f"Context for answering the question:\n{retrieved_context}\nQuestion/user input:\n{user_prompt}"
    # print(f"\n********\n{formatted_prompt}\n********\n")
    threadId = f"{uuid.uuid4()}"
    config = {"configurable": {"thread_id": threadId}}
    ai_response = model.invoke(messages[:-1] + [HumanMessage(content=formatted_prompt)], config=config)
    print(ai_response)

    return ai_response

def main():
    st.title("Chat with Nemesis Rules")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="""
            You are a personal assistant who answers questions about the rules for the game Nemesis.
            Only use the provided rule document to answer questions.
            Provide quotes from the document to support your answers.
            """)
        ]

    for message in st.session_state.messages:
        message_json = json.loads(message.json())
        message_type = message_json["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message_json["content"])

    if prompt := st.chat_input("What questions do you have?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            ai_response = prompt_ai(db, model, st.session_state.messages)
            st.markdown(ai_response.content)

        st.session_state.messages.append(ai_response)

if __name__ == "__main__":
    main()
