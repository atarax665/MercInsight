from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate

loader = DirectoryLoader("./docs", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                          chunk_overlap=50)

texts = splitter.split_documents(documents)

print(texts[-5: -1])
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={'normalize_embeddings': False},
    model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss")



