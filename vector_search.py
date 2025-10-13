import numpy as np
from google import genai
from google.genai import types
import os
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('api_key')
client = genai.Client(api_key=api_key)

def connect(mongo_client):
    client = mongo_client("mongodb+srv://dbVireyaVector:SenhaForteFrodoInutil@clustervireya.586jhai.mongodb.net/?retryWrites=true&w=majority&appName=Clustervireya")
    db = client["interVireya"]
    return db
db = connect(mongo_client=MongoClient)

def gerar_embeddings(text_list):
    """Gera embeddings normalizados para uma lista de textos."""
    if not text_list:
        return []
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text_list,
        config=types.EmbedContentConfig(output_dimensionality=512)
    )

    normed_embeddings = []
    for embedding_obj in result.embeddings:
        embedding_values_np = np.array(embedding_obj.values)
        normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)
        normed_embeddings.append(normed_embedding)
    
    return normed_embeddings

def vector_search_mongo(query_vector):
    collection = db["collection_Q&A_vectors"]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "embedding_index",       
                "path": "embedding",           
                "queryVector": query_vector,    
                "numCandidates": 1000,           
                "limit": 3                      
            }
        },
        {
            "$project": {
                "question": 1,
                "answer": 1,
                "score": {"$meta": "vectorSearchScore"} 
            }
        }
    ]

    results = list(collection.aggregate(pipeline))
    return results


def buscar_similares(query, k=3):
    """
    Recebe um texto, gera embedding normalizado usando gerar_embeddings,
    e retorna os top k documentos mais similares do MongoDB.
    """
    # 1️⃣ Gera embedding normalizado usando a função existente
    [query_embedding] = gerar_embeddings([query])  # retorna uma lista, pegamos o primeiro
    
    # 2️⃣ Chama a função de busca vetorial no MongoDB
    results = vector_search_mongo(query_embedding.tolist())  # converte np.array -> lista
    return results
