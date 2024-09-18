import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
import uuid

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from chainlit.types import AskFileResponse

import chainlit as cl
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
And if the user greets with greetings like Hi, hello, How are you, etc reply accordingly as well.
Example of your response should be:
The answer is foo
SOURCES: xyz
Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def generate_vdb(chunks=None):
    EMBEDDING_MODEL = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    PERSIST_PATH = "./qdrant_vector_db"  # Directory to store Qdrant collection
    COLLECTION_NAME = "legal_data"
    VECTOR_SIZE = 1536

    # Check if the vector database already exists
    if os.path.exists(PERSIST_PATH):
        print(f"Loading existing Qdrant database from {PERSIST_PATH}")
        qdrant_client = QdrantClient(path=PERSIST_PATH)  # Load the existing DB
        qdrant_vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
    else:
        print(f"Creating new Qdrant database at {PERSIST_PATH}")
        qdrant_client = QdrantClient(path=PERSIST_PATH)  # Create a new DB
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        qdrant_vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
        qdrant_vector_store.add_documents(chunks)
    return qdrant_vector_store


@cl.on_chat_start
async def on_chat_start():
    await cl.Avatar(
        name="Chat Legal AI",
        path="./chat_logo.jpg",
    ).send()

    pdf_links = [
    "https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf",
    "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"]

    if not os.path.exists("./qdrant_vector_db"):
        documents = []
        for pdf_link in pdf_links:
            loader = PyMuPDFLoader(pdf_link)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)

        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        split_chunks = text_splitter.split_documents(documents)

        docsearch = generate_vdb(split_chunks)
    else:
        docsearch = generate_vdb()

    # Let the user know that the system is ready
    msg = cl.Message(
        content=f"Welcome to the AI Legal Chatbot! Ask me anything about the AI policy", disable_human_feedback=True, author="Chat Legal AI"
        )
    await msg.send()

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements,author="Chat Legal AI").send()