import redis
import json
from datetime import datetime, timezone
import os
import psycopg2
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime, timezone
import json
from dotenv import load_dotenv
load_dotenv()

host = os.getenv('host')
porta = os.getenv('porta')
database = os.getenv('database')
senha = os.getenv('senha')
user = os.getenv('user')
def conectar():
    conn=psycopg2.connect(
        dbname=database,
        user=user,
        password=senha,
        host=host,
        port=porta
        )
    return conn

def connect_redis():
    r = redis.Redis(
        host=os.getenv('host_redis'),
        port=os.getenv('port_redis'),
        password=os.getenv('password'),
        ssl=True
    )
    return r


class RegistrarMemoriaArgs(BaseModel):
    session_id: str = Field(..., description="Identificador único da sessão do usuário.")
    content: Any = Field(..., description="Conteúdo ou dado que será armazenado na memória da sessão.")

@tool("registrar_memoria", args_schema=RegistrarMemoriaArgs)
def registrar_memoria(session_id: str, content: Any) -> str:
    """
    Registra uma entrada de memória associada a uma sessão específica no Redis.
    Cada memória inclui um timestamp UTC e o conteúdo fornecido.
    Retorna uma mensagem indicando sucesso ou falha.
    """
    key = f"memorys:{session_id}"
    timestamp = datetime.now(timezone.utc).isoformat()

    memory_entry = {
        "timestamp": timestamp,
        "data": content
    }

    try:
        r = connect_redis()
        r.rpush(key, json.dumps(memory_entry))
        return f"Memória registrada com sucesso na sessão '{session_id}'."
    except Exception as e:
        return f"Erro ao registrar memória: {e}"

class PopLastMemoryArgs(BaseModel):
    session_id: str = Field(..., description="Identificador único da sessão de onde a última memória será removida.")

@tool("pop_last_memory", args_schema=PopLastMemoryArgs)
def pop_last_memory(session_id: str) -> str:
    """
    Remove e retorna a última memória registrada na sessão especificada.
    Se não houver memórias armazenadas, retorna uma mensagem informando isso.
    """
    key = f"memorys:{session_id}"
    try:
        r = connect_redis()
        last = r.rpop(key)

        if last:
            return f"Última memória removida com sucesso: {last.decode('utf-8')}"
        else:
            return "Nenhuma memória encontrada para esta sessão."

    except Exception as e:
        return f"Erro ao remover última memória: {e}"

REDIS_TOOLS=[
    registrar_memoria,
    pop_last_memory
]