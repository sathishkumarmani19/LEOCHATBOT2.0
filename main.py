import os
import sys
import logging
from groq import Groq # Import Groq instead of GenAI
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chromadb.utils import embedding_functions

# --- RENDER SQLITE FIX ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass 

app = FastAPI()

# GREETING CONSTANT
EXACT_GREETING = (
    "Hello! I am Leo Bot, your HITS Expert. I'm delighted to provide you with "
    "detailed and professional information regarding Hindustan Institute of Technology "
    "and Science (HITS), particularly focusing on HITSEEE, Admissions, and the "
    "esteemed Department of Aeronautical Engineering."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GROQ INITIALIZATION ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Vector DB
try:
    db_client = chromadb.PersistentClient(path="./hits_vectordb")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db_client.get_collection(name="hits_web_data", embedding_function=default_ef)
except Exception as e:
    print(f"DB Error: {e}")

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        clean_query = query.text.lower().strip()
        
        # 1. Greeting Case
        if clean_query in ["hi", "hello", "hey", "start"]:
            return {"response": EXACT_GREETING}

        # 2. Search Database
        results = collection.query(
            query_texts=[clean_query], 
            n_results=3, # Reduced to 3 for faster response
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        if best_distance < 1.7:
            context = "\n".join(results['documents'][0])
            
            # 3. Ask Groq (Llama 3.3 70B)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are Leo Bot, the HITS Expert. {EXACT_GREETING} Use this context: {context}"
                    },
                    {
                        "role": "user",
                        "content": query.text,
                    }
                ],
                model="llama-3.3-70b-versatile", # The best model on Groq
                temperature=0.2, # Keeps answers professional
            )
            return {"response": chat_completion.choices[0].message.content}
        
        else:
            return {"response": "I don't have that specific info. Contact **info@hindustanuniv.ac.in**."}

    except Exception as e:
        return {"response": f"System Busy. Please email info@hindustanuniv.ac.in."}
