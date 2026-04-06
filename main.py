import os
import sys
import logging

# --- RENDER SQLITE VERSION FIX ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass 

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai 
from chromadb.utils import embedding_functions

os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

app = FastAPI()

# GREETING CONSTANT
EXACT_GREETING = (
    "Hello! I am Leo Bot, your HITS Expert. I'm delighted to provide you with "
    "detailed and professional information regarding Hindustan Institute of Technology "
    "and Science (HITS), particularly focusing on HITSEEE, Admissions, and the "
    "esteemed Department of Aeronautical Engineering."
)

# SETUP LOGGING FOR DEBUGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HITS_LEO_BOT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- TWO API KEY INITIALIZATION ---
# Make sure to set these names in your Render Environment Variables
api_keys = [
    os.getenv("GOOGLE_API_KEY_PRIMARY"),
    os.getenv("GOOGLE_API_KEY_SECONDARY")
]

# Create a list of clients (filters out None if a key is missing)
clients = [
    genai.Client(api_key=key, http_options={'api_version': 'v1beta'}) 
    for key in api_keys if key
]

try:
    db_client = chromadb.PersistentClient(path="./hits_vectordb")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db_client.get_collection(name="hits_knowledge", embedding_function=default_ef)
except Exception as e:
    logger.error(f"DB Error: {e}")

class Query(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"status": "online", "bot": "HITS Leo Bot 2.0 Failover Edition"}

@app.post("/chat")
async def chat(query: Query):
    try:
        clean_query = query.text.lower().strip()
        
        # 1. STRICT GREETING RULE
        if clean_query in ["hi", "hello", "hey", "start", "greetings"]:
            return {"response": EXACT_GREETING}

        # 2. SEARCH DATABASE
        results = collection.query(
            query_texts=[clean_query], 
            n_results=5, 
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        if best_distance < 1.7:
            context = "\n".join(results['documents'][0])
            persona_prefix = (
                f"You are Leo Bot, the HITS Expert. {EXACT_GREETING}\n\n"
                "INSTRUCTIONS: Use the following context to answer. "
                "Always use markdown tables for data. "
                f"Context: {context}"
            )
            full_prompt = f"{persona_prefix}\n\nUser Question: {query.text}"
        else:
            return {
                "response": "I'm sorry, I don't have that specific information in my context. Please contact **info@hindustanuniv.ac.in**."
            }

        
        # 3. DUAL-CLIENT & STABLE MODEL LOOP
        # We put 1.5-flash FIRST because it has a MUCH higher quota than 2.0
        model_priority = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]

        for i, current_client in enumerate(clients):
            for model_id in model_priority:
                try:
                    logger.info(f"Attempting Client {i+1} with {model_id}")
                    response = current_client.models.generate_content(
                        model=model_id,
                        contents=full_prompt
                    )
                    
                    if response and response.text:
                        return {"response": response.text}
                        
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Client {i+1} - {model_id} failed: {error_msg}")
                    
                    # 429 is the "Resource Exhausted" error (Wait 1 sec and try next key)
                    if "429" in error_msg:
                        time.sleep(1) 
                    
                    # 404 is the "Not Found" error (Continue to next model)
                    continue

        return {"response": "The HITS system is currently busy (All API keys exhausted). Please contact **info@hindustanuniv.ac.in**."}

    except Exception as final_e:
        logger.error(f"System Error: {final_e}")
        return {"response": f"System error: {str(final_e)}. Please contact **info@hindustanuniv.ac.in**."}
