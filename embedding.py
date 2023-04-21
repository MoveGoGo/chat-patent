import pickle
import os

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document

# os.environ["OPENAI_API_KEY"] = ""


# 将文件中的文字拆分成小的文件Embedding并持久化
def persist_embedding(text):
    documents = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # 将embedding数据持久化到本地磁盘
    embedding = OpenAIEmbeddings()
    # embedding = HuggingFaceInstructEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
