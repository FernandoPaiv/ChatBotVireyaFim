import redis
import json
from datetime import datetime, timezone
import os
import psycopg2
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

def get_session_id(email):
    try:
        conn=conectar()
        cursor=conn.cursor()
        cursor.execute(f'''SELECT id_funcionario FROM funcionario WHERE email='{email}' ''')
        dados=cursor.fetchone()
    except Exception as e:
        conn.rollback()
    return dados[0]

def get_memories(session_id):
    key=f'memorys:{session_id}'
    try:
        r = connect_redis()
        valores = r.lrange(key, 0, -1)
        valores = [v.decode('utf-8') for v in valores]
        memoria_objetos = []
        for v in valores:
            try:
                memoria_objetos.append(json.loads(v))
            except json.JSONDecodeError:
                memoria_objetos.append(v)
        return memoria_objetos
    except Exception as e:
        return 0