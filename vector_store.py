import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import config

def embedding_model():
    return SentenceTransformerEmbeddings(model_name=config.EMBED_MODEL)

def vector_store(docs=None, embeddings=None, db_p=config.DB_PATH, create_new=False):
    if os.path.exists(db_p) and not create_new:
        print(f"Loading existing DB from: {db_p}")
        if embeddings is None:
            embeddings = embedding_model()
        return Chroma(persist_directory=db_p, embedding_function=embeddings)
    
    elif docs and embeddings:
        print(f"Creating new DB at: {db_p}")
        vs = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=db_p)
        vs.persist()
        return vs

    if create_new and (docs is None or embeddings is None):
        raise ValueError("Need documents and embeddings to create a new vector store.")
    
    if not create_new and not os.path.exists(db_p):
        raise FileNotFoundError(f"Vector store not found at {db_p} and not creating a new one without documents.")
    
    raise ValueError("Invalid arguments for get_vector_store.")
