import chromadb
from chromadb.utils import embedding_functions
import os

# 1. Setup Chroma
db_client = chromadb.PersistentClient(path="./hits_vectordb")
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Delete old collection to start fresh
try:
    db_client.delete_collection(name="hits_knowledge")
except:
    pass

collection = db_client.create_collection(
    name="hits_knowledge", 
    embedding_function=default_ef
)

# 2. Function to load and "Label" data
def prepare_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by your [LABEL] tags
    chunks = content.split("[LABEL:")[1:] 
    
    documents = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        label_end = chunk.find("]")
        label = chunk[:label_end]
        text = chunk[label_end+1:].strip()
        
        documents.append(text)
        metadatas.append({"source": "hits_manual", "category": label})
        ids.append(f"id_{i}")
        
    return documents, metadatas, ids

# 3. Execute the Build
docs, meta, ids = prepare_data("hits_data.txt")
collection.add(
    documents=docs,
    metadatas=meta,
    ids=ids
)

print(f"Successfully indexed {len(docs)} HITS knowledge chunks!")